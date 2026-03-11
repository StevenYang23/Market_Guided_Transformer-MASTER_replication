"""
Backtest Framework
Strategy: Daily top-K stocks selection based on predicted return score.
Metrics:
  - AR (Excess Annualized Return): annualized mean daily excess return over benchmark
  - IR (Information Ratio): mean(daily_excess) / std(daily_excess) * sqrt(252)
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import os
import torch
from torch.utils.data import DataLoader

from base_model import DailyBatchSamplerRandom
from master      import MASTERModel
from transformer import TransformerModel
from gat         import GATModel
from dtml        import DTMLModel


# ════════════════════════════════════════════════════════════════════════════
# 1. Build model and load saved weights
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_type, universe, prefix, seed, d_model=256, dropout=0.5, GPU=0):
    """
    Instantiate the specified model and load pre-trained weights.
    Weight file naming convention: model/{universe}_{prefix}_{model_type}_{seed}.pkl
    """
    common = dict(
        n_epochs=1, lr=1e-5, GPU=GPU, seed=seed,
        train_stop_loss_thred=None,
        save_path='model',
        save_prefix=f'{universe}_{prefix}_{model_type}',
    )

    if model_type == 'master':
        model = MASTERModel(
            d_feat=158, d_model=d_model, t_nhead=4, s_nhead=2,
            T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=(5 if universe == 'csi300' else 2),
            gate_input_start_index=158,
            gate_input_end_index=221,
            **common,
        )
        # master uses a different naming convention (no model_type in filename)
        param_path = f'model/{universe}_{prefix}_{seed}.pkl'

    elif model_type == 'transformer':
        model = TransformerModel(
            d_feat=158, d_model=d_model, nhead=4, num_layers=2,
            dim_feedforward=512, dropout=dropout, **common,
        )
        param_path = f'model/{universe}_{prefix}_transformer_{seed}.pkl'

    elif model_type == 'gat':
        model = GATModel(
            d_feat=158, d_model=d_model, gat_hidden=64,
            nhead=4, gru_layers=2, dropout=dropout, **common,
        )
        param_path = f'model/{universe}_{prefix}_gat_{seed}.pkl'

    elif model_type == 'dtml':
        model = DTMLModel(
            d_feat=158, d_model=d_model,
            nhead_temporal=4, nhead_stock=4,
            num_temporal_layers=2, dim_feedforward=512,
            dropout=dropout, **common,
        )
        param_path = f'model/{universe}_{prefix}_dtml_{seed}.pkl'

    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    print(f'  Loading weights from {param_path}')
    model.load_param(param_path)
    return model


# ════════════════════════════════════════════════════════════════════════════
# 2. Extract ground-truth labels from dl_test
# ════════════════════════════════════════════════════════════════════════════

def extract_label_df(dl_test) -> pd.DataFrame:
    """
    Reconstruct the true label DataFrame from dl_test.
    Each sample's label is data[:, -1, -1] (last time step, last feature = return).

    Returns
    -------
    label_df : pd.DataFrame, shape (n_dates, n_stocks)
               index=datetime, columns=instrument
    """
    index = dl_test.get_index()   # MultiIndex (datetime, instrument)

    sampler = DailyBatchSamplerRandom(dl_test, shuffle=False)
    loader  = DataLoader(dl_test, sampler=sampler, drop_last=False)

    labels_all = []
    for data in loader:
        data = torch.squeeze(data, dim=0)   # (N, T, F)
        labels_all.append(data[:, -1, -1].numpy().ravel())

    label_series = pd.Series(np.concatenate(labels_all), index=index)
    label_df = label_series.unstack(level='instrument')
    label_df.index = pd.to_datetime(label_df.index)
    return label_df.sort_index()


# ════════════════════════════════════════════════════════════════════════════
# 3. Core backtest: compute AR and IR
# ════════════════════════════════════════════════════════════════════════════

def calc_ar_ir(
    predictions: pd.Series,
    label_df: pd.DataFrame,
    top_k: int = 30,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Simulate daily trading: each day select top-K stocks by predicted score,
    hold equal-weight portfolio, compute excess return over equal-weight benchmark.

    Parameters
    ----------
    predictions : pd.Series with MultiIndex (datetime, instrument)
    label_df    : pd.DataFrame, shape (n_dates, n_stocks), true next-period returns
    top_k       : number of stocks to hold each day
    trading_days_per_year : used for annualisation

    Returns
    -------
    dict with keys: AR, IR, daily_excess (pd.Series), cum_excess (pd.Series)
    """
    # Reshape predictions to DataFrame (n_dates, n_stocks)
    pred_df = predictions.unstack(level='instrument')
    pred_df.index = pd.to_datetime(pred_df.index)
    pred_df = pred_df.sort_index()

    dates = pred_df.index.intersection(label_df.index)
    pred_df  = pred_df.loc[dates]
    label_df = label_df.loc[dates]

    daily_excess = []
    valid_dates  = []

    for date in dates:
        scores = pred_df.loc[date].dropna()
        labels = label_df.loc[date].dropna()

        common = scores.index.intersection(labels.index)
        if len(common) < top_k:
            continue

        scores = scores[common]
        labels = labels[common]

        # Equal-weight top-K portfolio return
        selected  = scores.nlargest(top_k).index
        port_ret  = labels[selected].mean()

        # Equal-weight full-universe benchmark return
        bench_ret = labels.mean()

        daily_excess.append(port_ret - bench_ret)
        valid_dates.append(date)

    excess = np.array(daily_excess)

    # Annualized excess return
    AR = excess.mean() * trading_days_per_year

    # Information Ratio: annualized mean / annualized std of daily excess
    IR = (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(trading_days_per_year)

    daily_excess_s = pd.Series(excess, index=valid_dates)
    cum_excess_s   = (daily_excess_s + 1).cumprod() - 1

    return {
        'AR':          AR,
        'IR':          IR,
        'daily_excess': daily_excess_s,
        'cum_excess':   cum_excess_s,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Backtest: AR and IR for stock prediction models')
    parser.add_argument('--universe', type=str, default='csi300',
                        choices=['csi300', 'csi800'])
    parser.add_argument('--prefix', type=str, default='opensource',
                        choices=['original', 'opensource'])
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['master', 'transformer', 'gat', 'dtml'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='data/opensource')
    args = parser.parse_args()

    # Load test dataset
    test_path = f'{args.data_dir}/{args.universe}_dl_test.pkl'
    print(f'Loading test data from {test_path} ...')
    with open(test_path, 'rb') as f:
        dl_test = pickle.load(f)
    print('Test data loaded.')

    # Extract ground-truth labels once (shared across all seeds)
    print('Extracting ground-truth labels ...')
    label_df = extract_label_df(dl_test)
    print(f'  label_df shape: {label_df.shape}  '
          f'({label_df.index[0].date()} ~ {label_df.index[-1].date()})')

    # Output directory
    out_dir = f'backtest_results/{args.universe}_{args.model_type}'
    os.makedirs(out_dir, exist_ok=True)

    all_ar, all_ir = [], []

    for seed in args.seeds:
        print(f'\n{"="*55}')
        print(f' Seed {seed}  |  {args.model_type}  |  {args.universe}  |  top-{args.top_k}')
        print(f'{"="*55}')

        # Load model and run inference
        model = load_model(
            model_type=args.model_type,
            universe=args.universe,
            prefix=args.prefix,
            seed=seed,
            GPU=args.GPU,
        )
        predictions, _ = model.predict(dl_test)

        # Compute AR and IR
        result = calc_ar_ir(predictions, label_df, top_k=args.top_k)

        print(f'  AR = {result["AR"]:.4f}')
        print(f'  IR = {result["IR"]:.4f}')

        # Save cumulative excess return curve
        curve_path = f'{out_dir}/cum_excess_seed{seed}.csv'
        result['cum_excess'].to_csv(curve_path, header=['cum_excess_return'])
        print(f'  Cumulative excess return saved to {curve_path}')

        all_ar.append(result['AR'])
        all_ir.append(result['IR'])

    # Summary across seeds
    print(f'\n{"="*55}')
    print(f' Summary  [{args.model_type}  {args.universe}  top-{args.top_k}  '
          f'seeds={args.seeds}]')
    print(f'{"="*55}')
    print(f'  AR:  {np.mean(all_ar):.4f} ± {np.std(all_ar):.4f}')
    print(f'  IR:  {np.mean(all_ir):.4f} ± {np.std(all_ir):.4f}')

    # Save summary CSV
    summary = pd.DataFrame({
        'seed': args.seeds,
        'AR':   all_ar,
        'IR':   all_ir,
    })
    summary_path = f'{out_dir}/summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\n  Summary saved to {summary_path}')


if __name__ == '__main__':
    main()