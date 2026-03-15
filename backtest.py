"""
Backtest Framework
Strategy: Daily top-K stocks selection based on predicted return score.
Metrics:
  - AR (Excess Annualized Return): annualized mean daily excess return over benchmark
  - IR (Information Ratio): mean(daily_excess) / std(daily_excess) * sqrt(252)

Transaction cost model:
  - Commission: one-way fee charged on buy and sell turnover (default 0.05%)
  - Stamp duty: charged on sell side only in A-share market   (default 0.10%)
  - Slippage:   bid-ask spread cost charged on both sides     (default 0.05%)
  Total one-way cost  = commission + slippage
  Total round-trip    = 2*(commission + slippage) + stamp_duty  ≈ 0.25% per trade
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
    index = dl_test.get_index()

    sampler = DailyBatchSamplerRandom(dl_test, shuffle=False)
    loader  = DataLoader(dl_test, sampler=sampler, drop_last=False)

    labels_all = []
    for data in loader:
        data = torch.squeeze(data, dim=0)
        labels_all.append(data[:, -1, -1].numpy().ravel())

    label_series = pd.Series(np.concatenate(labels_all), index=index)
    label_df = label_series.unstack(level='instrument')
    label_df.index = pd.to_datetime(label_df.index)
    return label_df.sort_index()


# ════════════════════════════════════════════════════════════════════════════
# 3. Transaction cost model
# ════════════════════════════════════════════════════════════════════════════

def calc_transaction_cost(
    prev_holdings: set,
    curr_holdings: set,
    commission: float,
    stamp_duty: float,
    slippage: float,
) -> float:
    """
    Compute total transaction cost for one rebalance as a fraction of portfolio value.
    Assumes equal-weight portfolio: each stock weight = 1/top_k.

    Cost components (A-share market convention):
      - commission : charged on both buy and sell turnover
      - stamp_duty : charged on sell turnover only
      - slippage   : half bid-ask spread, charged on both buy and sell

    Parameters
    ----------
    prev_holdings : set of stock codes held yesterday
    curr_holdings : set of stock codes held today
    commission    : one-way commission rate  (e.g. 0.0005 = 0.05%)
    stamp_duty    : sell-side stamp duty rate (e.g. 0.001  = 0.10%)
    slippage      : one-way slippage rate    (e.g. 0.0005 = 0.05%)

    Returns
    -------
    cost : float, total cost as a fraction of portfolio NAV
    """
    if not prev_holdings:
        # First day: buy all positions (no sell cost)
        buy_turnover  = 1.0
        sell_turnover = 0.0
    else:
        n = len(curr_holdings)
        sold    = prev_holdings - curr_holdings   # stocks being sold
        bought  = curr_holdings - prev_holdings   # stocks being bought
        # Each stock has equal weight 1/n
        sell_turnover = len(sold)   / n
        buy_turnover  = len(bought) / n

    cost = (
        (buy_turnover  + sell_turnover) * (commission + slippage)   # two-way commission + slippage
        + sell_turnover * stamp_duty                                 # sell-side stamp duty only
    )
    return cost


# ════════════════════════════════════════════════════════════════════════════
# 4. Core backtest: compute AR and IR with transaction costs
# ════════════════════════════════════════════════════════════════════════════

def calc_ar_ir(
    predictions: pd.Series,
    label_df: pd.DataFrame,
    top_k: int = 30,
    trading_days_per_year: int = 252,
    commission: float = 0.0005,
    stamp_duty: float = 0.001,
    slippage: float   = 0.0005,
) -> dict:
    """
    Simulate daily trading with transaction costs and slippage.

    Default cost assumptions (A-share market):
      commission  = 0.05% one-way  (broker fee, negotiable, often lower)
      stamp_duty  = 0.10% sell-side (fixed by regulation)
      slippage    = 0.05% one-way  (conservative estimate for liquid CSI300 stocks)
      ─────────────────────────────────────────────────────
      Round-trip  ≈ 0.25% per full position turnover

    Parameters
    ----------
    predictions : pd.Series, MultiIndex (datetime, instrument)
    label_df    : pd.DataFrame (n_dates × n_stocks), true next-period returns
    top_k       : number of stocks held each day
    commission  : one-way commission rate
    stamp_duty  : sell-side stamp duty rate
    slippage    : one-way slippage rate
    """
    pred_df = predictions.unstack(level='instrument')
    pred_df.index = pd.to_datetime(pred_df.index)
    pred_df = pred_df.sort_index()

    dates = pred_df.index.intersection(label_df.index)
    pred_df  = pred_df.loc[dates]
    label_df = label_df.loc[dates]

    daily_excess_gross = []   # before costs
    daily_excess_net   = []   # after costs
    daily_cost         = []
    valid_dates        = []
    prev_holdings: set = set()

    for date in dates:
        scores = pred_df.loc[date].dropna()
        labels = label_df.loc[date].dropna()
        common = scores.index.intersection(labels.index)
        if len(common) < top_k:
            continue

        scores = scores[common]
        labels = labels[common]

        # Select top-K portfolio
        selected     = set(scores.nlargest(top_k).index)
        selected_list = list(selected)

        # Gross portfolio return (equal-weight)
        port_ret_gross = labels[selected_list].mean()

        # Equal-weight full-universe benchmark return
        bench_ret = labels.mean()

        # Transaction cost for today's rebalance
        cost = calc_transaction_cost(
            prev_holdings, selected,
            commission, stamp_duty, slippage,
        )

        # Net portfolio return after costs
        port_ret_net = port_ret_gross - cost

        daily_excess_gross.append(port_ret_gross - bench_ret)
        daily_excess_net.append(port_ret_net   - bench_ret)
        daily_cost.append(cost)
        valid_dates.append(date)
        prev_holdings = selected

    excess_gross = np.array(daily_excess_gross)
    excess_net   = np.array(daily_excess_net)
    costs        = np.array(daily_cost)

    def annualise(excess):
        AR = excess.mean() * trading_days_per_year
        IR = (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(trading_days_per_year)
        return AR, IR

    AR_gross, IR_gross = annualise(excess_gross)
    AR_net,   IR_net   = annualise(excess_net)

    excess_net_s   = pd.Series(excess_net,   index=valid_dates)
    excess_gross_s = pd.Series(excess_gross, index=valid_dates)
    cost_s         = pd.Series(costs,        index=valid_dates)

    return {
        # Net (after cost) — main result
        'AR':  AR_net,
        'IR':  IR_net,
        # Gross (before cost) — for reference
        'AR_gross': AR_gross,
        'IR_gross': IR_gross,
        # Details
        'daily_excess_net':   excess_net_s,
        'daily_excess_gross': excess_gross_s,
        'cum_excess_net':     (excess_net_s   + 1).cumprod() - 1,
        'cum_excess_gross':   (excess_gross_s + 1).cumprod() - 1,
        'daily_cost':         cost_s,
        'avg_daily_cost':     costs.mean(),
        'avg_annual_cost':    costs.mean() * trading_days_per_year,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Backtest: AR and IR for stock prediction models')
    parser.add_argument('--universe',    type=str,   default='csi300',
                        choices=['csi300', 'csi800'])
    parser.add_argument('--prefix',      type=str,   default='opensource',
                        choices=['original', 'opensource'])
    parser.add_argument('--model_type',  type=str,   default='transformer',
                        choices=['master', 'transformer', 'gat', 'dtml'])
    parser.add_argument('--seeds',       type=int,   nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--top_k',       type=int,   default=30)
    parser.add_argument('--GPU',         type=int,   default=0)
    parser.add_argument('--data_dir',    type=str,   default='data/opensource')
    # Transaction cost parameters
    parser.add_argument('--commission',  type=float, default=0.0005,
                        help='One-way commission rate (default: 0.05%%)')
    parser.add_argument('--stamp_duty',  type=float, default=0.001,
                        help='Sell-side stamp duty rate (default: 0.10%%)')
    parser.add_argument('--slippage',    type=float, default=0.002,
                        help='One-way slippage rate (default: 0.05%%)')
    args = parser.parse_args()

    # Print cost assumptions
    round_trip = 2*(args.commission + args.slippage) + args.stamp_duty
    print(f'\nTransaction cost assumptions:')
    print(f'  Commission (one-way) : {args.commission*100:.3f}%')
    print(f'  Stamp duty (sell)    : {args.stamp_duty*100:.3f}%')
    print(f'  Slippage   (one-way) : {args.slippage*100:.3f}%')
    print(f'  Round-trip total     : {round_trip*100:.3f}%')

    # Load test dataset
    test_path = f'{args.data_dir}/{args.universe}_dl_test.pkl'
    print(f'\nLoading test data from {test_path} ...')
    with open(test_path, 'rb') as f:
        dl_test = pickle.load(f)
    print('Test data loaded.')

    # Extract ground-truth labels once (shared across all seeds)
    print('Extracting ground-truth labels ...')
    label_df = extract_label_df(dl_test)
    print(f'  label_df shape: {label_df.shape}  '
          f'({label_df.index[0].date()} ~ {label_df.index[-1].date()})')

    out_dir = f'backtest_results/{args.universe}_{args.model_type}'
    os.makedirs(out_dir, exist_ok=True)

    all_ar, all_ir = [], []
    all_ar_gross, all_ir_gross = [], []

    for seed in args.seeds:
        print(f'\n{"="*55}')
        print(f' Seed {seed} | {args.model_type} | {args.universe} | top-{args.top_k}')
        print(f'{"="*55}')

        model = load_model(
            model_type=args.model_type,
            universe=args.universe,
            prefix=args.prefix,
            seed=seed,
            GPU=args.GPU,
        )
        predictions, _ = model.predict(dl_test)

        result = calc_ar_ir(
            predictions, label_df,
            top_k=args.top_k,
            commission=args.commission,
            stamp_duty=args.stamp_duty,
            slippage=args.slippage,
        )

        print(f'  AR  (net)   = {result["AR"]:.4f}    IR  (net)   = {result["IR"]:.4f}')
        print(f'  AR  (gross) = {result["AR_gross"]:.4f}    IR  (gross) = {result["IR_gross"]:.4f}')
        print(f'  Avg daily cost = {result["avg_daily_cost"]*100:.4f}%  '
              f'| Annualised cost drag = {result["avg_annual_cost"]*100:.2f}%')

        # Save curves
        curve_df = pd.DataFrame({
            'cum_excess_net':   result['cum_excess_net'],
            'cum_excess_gross': result['cum_excess_gross'],
            'daily_cost':       result['daily_cost'],
        })
        curve_path = f'{out_dir}/cum_excess_seed{seed}.csv'
        curve_df.to_csv(curve_path)
        print(f'  Curves saved to {curve_path}')

        all_ar.append(result['AR'])
        all_ir.append(result['IR'])
        all_ar_gross.append(result['AR_gross'])
        all_ir_gross.append(result['IR_gross'])

    # Summary
    print(f'\n{"="*55}')
    print(f' Summary [{args.model_type}  {args.universe}  top-{args.top_k}]')
    print(f'{"="*55}')
    print(f'  AR  (net):   {np.mean(all_ar):.4f} ± {np.std(all_ar):.4f}')
    print(f'  IR  (net):   {np.mean(all_ir):.4f} ± {np.std(all_ir):.4f}')
    print(f'  AR  (gross): {np.mean(all_ar_gross):.4f} ± {np.std(all_ar_gross):.4f}')
    print(f'  IR  (gross): {np.mean(all_ir_gross):.4f} ± {np.std(all_ir_gross):.4f}')

    summary = pd.DataFrame({
        'seed':     args.seeds,
        'AR_net':   all_ar,
        'IR_net':   all_ir,
        'AR_gross': all_ar_gross,
        'IR_gross': all_ir_gross,
    })
    summary_path = f'{out_dir}/summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\n  Summary saved to {summary_path}')


if __name__ == '__main__':
    main()