"""
Qlib-based backtest for MASTER and baselines.
Uses Qlib PortAnaRecord to compute AR, IR and other portfolio metrics
with the official CSI300 / CSI800 benchmark index.
"""

import pickle
import numpy as np
import pandas as pd
import argparse

import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.backtest import backtest, executor as exec_cls
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy

from master      import MASTERModel
from transformer import TransformerModel
from gat         import GATModel
from dtml        import DTMLModel


# ════════════════════════════════════════════════════════════════════════════
# 1. Init Qlib
# ════════════════════════════════════════════════════════════════════════════

def init_qlib(qlib_data_path: str = '~/.qlib/qlib_data/cn_data'):
    qlib.init(provider_uri=qlib_data_path, region=REG_CN)


# ════════════════════════════════════════════════════════════════════════════
# 2. Load model and run inference (same as backtest.py)
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_type, universe, prefix, seed, d_model=256, dropout=0.5, GPU=0):
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
            gate_input_start_index=158, gate_input_end_index=221,
            **common,
        )
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
# 3. Convert predictions to Qlib signal format
# ════════════════════════════════════════════════════════════════════════════

def predictions_to_signal(predictions: pd.Series) -> pd.DataFrame:
    """
    Convert model output pd.Series (MultiIndex: datetime x instrument)
    to Qlib-compatible signal DataFrame with column 'score'.
    """
    signal = predictions.to_frame('score')
    signal.index.names = ['datetime', 'instrument']
    signal = signal.sort_index()
    return signal


# ════════════════════════════════════════════════════════════════════════════
# 4. Get benchmark return from Qlib
# ════════════════════════════════════════════════════════════════════════════

def get_benchmark_return(universe: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch daily benchmark index close price from Qlib and compute daily return.
    CSI300 -> SH000300, CSI800 -> SH000906
    """
    benchmark_map = {
        'csi300': 'SH000300',
        'csi800': 'SH000906',
    }
    index_code = benchmark_map[universe]

    # Fetch close price from Qlib
    bench_df = D.features(
        instruments=[index_code],
        fields=['$close'],
        start_time=start_date,
        end_time=end_date,
        freq='day',
    )
    bench_close = bench_df['$close'].unstack(level='instrument')[index_code]
    bench_ret   = bench_close.pct_change().dropna()
    bench_ret.name = 'bench_return'
    return bench_ret


# ════════════════════════════════════════════════════════════════════════════
# 5. Qlib portfolio backtest
# ════════════════════════════════════════════════════════════════════════════

def run_qlib_backtest(
    signal: pd.DataFrame,
    bench_ret: pd.Series,
    top_k: int = 30,
    n_drop: int = 5,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Use Qlib TopkDropoutStrategy to simulate portfolio,
    then compute AR and IR against the benchmark.

    Parameters
    ----------
    signal    : DataFrame with MultiIndex (datetime, instrument) and column 'score'
    bench_ret : pd.Series of daily benchmark returns, index=datetime
    top_k     : number of stocks to hold
    n_drop    : number of stocks replaced each day (set 0 for pure top-K)
    """
    # Build daily portfolio weights using top-K selection
    dates = signal.index.get_level_values('datetime').unique().sort_values()

    port_rets = []
    valid_dates = []

    for date in dates:
        day_signal = signal.loc[date, 'score'].dropna()
        if len(day_signal) < top_k:
            continue
        selected = day_signal.nlargest(top_k).index.tolist()

        # Fetch next-day actual return from Qlib
        try:
            next_idx = dates[dates.get_loc(date) + 1]
        except (IndexError, KeyError):
            continue

        ret_df = D.features(
            instruments=selected,
            fields=['$close'],
            start_time=date,
            end_time=next_idx,
            freq='day',
        )
        if ret_df.empty:
            continue

        close = ret_df['$close'].unstack('instrument')
        if len(close) < 2:
            continue

        daily_ret = close.iloc[-1] / close.iloc[0] - 1
        port_ret  = daily_ret.mean()

        port_rets.append(port_ret)
        valid_dates.append(next_idx)

    port_series  = pd.Series(port_rets, index=valid_dates)
    bench_aligned = bench_ret.reindex(valid_dates).fillna(0)
    excess        = port_series - bench_aligned

    AR = excess.mean() * trading_days_per_year
    IR = (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(trading_days_per_year)

    cum_excess = (excess + 1).cumprod() - 1

    return {
        'AR':           AR,
        'IR':           IR,
        'daily_excess': excess,
        'cum_excess':   cum_excess,
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. Lightweight fallback: use label col as return (no Qlib price needed)
# ════════════════════════════════════════════════════════════════════════════

def run_label_backtest(
    predictions: pd.Series,
    dl_test,
    universe: str,
    top_k: int = 30,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Fallback backtest using the label column in dl_test as true returns,
    and the CSI300/CSI800 equal-weight universe mean as benchmark.
    This avoids fetching price data from Qlib.
    """
    import torch
    from torch.utils.data import DataLoader
    from base_model import DailyBatchSamplerRandom

    # Reconstruct label series from dl_test
    index   = dl_test.get_index()
    sampler = DailyBatchSamplerRandom(dl_test, shuffle=False)
    loader  = DataLoader(dl_test, sampler=sampler, drop_last=False)

    labels_all = []
    for data in loader:
        data = torch.squeeze(data, dim=0)
        labels_all.append(data[:, -1, -1].numpy().ravel())

    label_series = pd.Series(np.concatenate(labels_all), index=index)

    # Align predictions and labels
    pred_df  = predictions.unstack('instrument')
    label_df = label_series.unstack('instrument')
    pred_df.index  = pd.to_datetime(pred_df.index)
    label_df.index = pd.to_datetime(label_df.index)

    dates = pred_df.index.intersection(label_df.index)

    daily_excess = []
    valid_dates  = []

    for date in dates:
        scores = pred_df.loc[date].dropna()
        labels = label_df.loc[date].dropna()
        common = scores.index.intersection(labels.index)
        if len(common) < top_k:
            continue

        selected  = scores[common].nlargest(top_k).index
        port_ret  = labels[common][selected].mean()   # portfolio
        bench_ret = labels[common].mean()             # equal-weight universe

        daily_excess.append(port_ret - bench_ret)
        valid_dates.append(date)

    excess     = np.array(daily_excess)
    AR         = excess.mean() * trading_days_per_year
    IR         = (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(trading_days_per_year)
    excess_s   = pd.Series(excess, index=valid_dates)
    cum_excess = (excess_s + 1).cumprod() - 1

    return {'AR': AR, 'IR': IR, 'daily_excess': excess_s, 'cum_excess': cum_excess}


# ════════════════════════════════════════════════════════════════════════════
# 7. Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe',    type=str, default='csi300',
                        choices=['csi300', 'csi800'])
    parser.add_argument('--prefix',      type=str, default='opensource')
    parser.add_argument('--model_type',  type=str, default='transformer',
                        choices=['master', 'transformer', 'gat', 'dtml'])
    parser.add_argument('--seeds',       type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--top_k',       type=int, default=30)
    parser.add_argument('--GPU',         type=int, default=0)
    parser.add_argument('--data_dir',    type=str, default='data/opensource')
    parser.add_argument('--qlib_data',   type=str, default='~/.qlib/qlib_data/cn_data',
                        help='Path to Qlib data for fetching benchmark index')
    parser.add_argument('--use_qlib_bench', action='store_true',
                        help='Use real index price from Qlib as benchmark (requires Qlib data)')
    args = parser.parse_args()

    # Load test pkl
    test_path = f'{args.data_dir}/{args.universe}_dl_test.pkl'
    print(f'Loading test data from {test_path} ...')
    with open(test_path, 'rb') as f:
        dl_test = pickle.load(f)
    print('Test data loaded.')

    # Optionally init Qlib and fetch real benchmark
    bench_ret = None
    if args.use_qlib_bench:
        print('Initialising Qlib ...')
        init_qlib(args.qlib_data)
        index      = dl_test.get_index()
        start_date = str(index.get_level_values('datetime').min().date())
        end_date   = str(index.get_level_values('datetime').max().date())
        print(f'Fetching benchmark ({args.universe}) from {start_date} to {end_date} ...')
        bench_ret = get_benchmark_return(args.universe, start_date, end_date)
        print('Benchmark loaded.')

    out_dir = f'backtest_results/{args.universe}_{args.model_type}'
    os.makedirs(out_dir, exist_ok=True)

    all_ar, all_ir = [], []

    for seed in args.seeds:
        print(f'\n{"="*55}')
        print(f' Seed {seed} | {args.model_type} | {args.universe} | top-{args.top_k}')
        print(f'{"="*55}')

        model       = load_model(args.model_type, args.universe, args.prefix, seed, GPU=args.GPU)
        predictions, _ = model.predict(dl_test)

        if args.use_qlib_bench and bench_ret is not None:
            # Use real CSI300/CSI800 index as benchmark
            signal = predictions_to_signal(predictions)
            result = run_qlib_backtest(signal, bench_ret, top_k=args.top_k)
        else:
            # Fallback: use equal-weight universe mean as benchmark
            result = run_label_backtest(predictions, dl_test, args.universe, top_k=args.top_k)

        print(f'  AR = {result["AR"]:.4f}')
        print(f'  IR = {result["IR"]:.4f}')

        curve_path = f'{out_dir}/cum_excess_seed{seed}.csv'
        result['cum_excess'].to_csv(curve_path, header=['cum_excess_return'])

        all_ar.append(result['AR'])
        all_ir.append(result['IR'])

    print(f'\n{"="*55}')
    print(f' Summary [{args.model_type}  {args.universe}  top-{args.top_k}]')
    print(f'{"="*55}')
    print(f'  AR:  {np.mean(all_ar):.4f} ± {np.std(all_ar):.4f}')
    print(f'  IR:  {np.mean(all_ir):.4f} ± {np.std(all_ir):.4f}')

    summary = pd.DataFrame({'seed': args.seeds, 'AR': all_ar, 'IR': all_ir})
    summary.to_csv(f'{out_dir}/summary.csv', index=False)
    print(f'  Results saved to {out_dir}/summary.csv')


if __name__ == '__main__':
    import os
    main()