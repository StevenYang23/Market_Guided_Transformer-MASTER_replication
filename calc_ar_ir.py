"""
AR/IR 回测计算模块
基于已有预测分数(scores)计算年化超额收益(AR)和信息比率(IR)
与 backtest.py 保持一致的计算逻辑
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import warnings
import argparse
import os

warnings.filterwarnings("ignore")

# 可视化
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calc_ar_ir(
    predictions: pd.Series,
    label_df: pd.DataFrame,
    top_k: int = 30,
    trading_days_per_year: int = 252,
    commission: float = 0.0005,
    stamp_duty: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
    """
    计算AR和IR - 与 backtest.py 保持一致的逻辑

    Parameters
    ----------
    predictions : pd.Series with MultiIndex (datetime, instrument)
    label_df    : pd.DataFrame, shape (n_dates, n_stocks), true next-period returns
    top_k       : number of stocks to hold each day
    trading_days_per_year : used for annualisation
    commission  : 佣金费率 (单边), default 0.05%
    stamp_duty  : 印花税费率 (仅卖出), default 0.10%
    slippage    : 滑点费率 (双边), default 0.05%

    Returns
    -------
    dict with keys: AR, IR, daily_excess, cum_excess, etc.
    """
    # 计算交易成本
    one_way_cost = commission + slippage
    round_trip_cost = 2 * one_way_cost + stamp_duty
    # Reshape predictions to DataFrame (n_dates, n_stocks)
    pred_df = predictions.unstack(level='instrument')
    pred_df.index = pd.to_datetime(pred_df.index)
    pred_df = pred_df.sort_index()

    dates = pred_df.index.intersection(label_df.index)
    pred_df = pred_df.loc[dates]
    label_df = label_df.loc[dates]

    daily_excess = []
    daily_excess_net = []
    valid_dates = []
    portfolio_returns = []
    benchmark_returns = []
    daily_turnover = []
    daily_trading_cost = []

    prev_holdings = set()

    for date in dates:
        scores = pred_df.loc[date].dropna()
        labels = label_df.loc[date].dropna()

        common = scores.index.intersection(labels.index)
        if len(common) < top_k:
            continue

        scores = scores[common]
        labels = labels[common]

        # Equal-weight top-K portfolio return
        selected = set(scores.nlargest(top_k).index)
        port_ret = labels[list(selected)].mean()

        # Equal-weight full-universe benchmark return
        bench_ret = labels.mean()

        # 计算换手率
        if prev_holdings:
            # 调出的股票数 / top_k
            turnover = len(selected - prev_holdings) / top_k
        else:
            turnover = 1.0  # 首日建仓，全部买入

        # 计算交易成本
        trading_cost = turnover * round_trip_cost
        net_excess = (port_ret - bench_ret) - trading_cost

        daily_excess.append(port_ret - bench_ret)
        daily_excess_net.append(net_excess)
        portfolio_returns.append(port_ret)
        benchmark_returns.append(bench_ret)
        daily_turnover.append(turnover)
        daily_trading_cost.append(trading_cost)
        valid_dates.append(date)

        prev_holdings = selected

    excess = np.array(daily_excess)
    excess_net = np.array(daily_excess_net)

    # Annualized excess return (gross, no cost)
    AR = excess.mean() * trading_days_per_year
    IR = (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(trading_days_per_year)

    # Annualized excess return (net, with cost)
    AR_net = excess_net.mean() * trading_days_per_year
    IR_net = (excess_net.mean() / (excess_net.std() + 1e-8)) * np.sqrt(trading_days_per_year)

    # Other statistics
    annualized_return = np.mean(portfolio_returns) * trading_days_per_year
    annualized_vol = np.std(portfolio_returns) * np.sqrt(trading_days_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    cum_ret = (1 + np.array(portfolio_returns)).cumprod()
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    max_drawdown = drawdown.min()

    daily_excess_s = pd.Series(excess, index=valid_dates)
    cum_excess_s = (daily_excess_s + 1).cumprod() - 1

    daily_excess_net_s = pd.Series(excess_net, index=valid_dates)
    cum_excess_net_s = (daily_excess_net_s + 1).cumprod() - 1

    # 换手率统计
    turnover_s = pd.Series(daily_turnover, index=valid_dates)
    trading_cost_s = pd.Series(daily_trading_cost, index=valid_dates)

    return {
        'AR': AR,
        'IR': IR,
        'AR_net': AR_net,
        'IR_net': IR_net,
        'daily_excess': daily_excess_s,
        'cum_excess': cum_excess_s,
        'daily_excess_net': daily_excess_net_s,
        'cum_excess_net': cum_excess_net_s,
        'daily_turnover': turnover_s,
        'daily_trading_cost': trading_cost_s,
        'avg_turnover': turnover_s.mean(),
        'total_trading_cost': trading_cost_s.sum(),
        'annualized_trading_cost': trading_cost_s.mean() * trading_days_per_year,
        'daily_portfolio_return': pd.Series(portfolio_returns, index=valid_dates),
        'daily_benchmark_return': pd.Series(benchmark_returns, index=valid_dates),
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': (excess > 0).mean(),
        'win_rate_net': (excess_net > 0).mean(),
        'n_days': len(excess),
        'commission': commission,
        'stamp_duty': stamp_duty,
        'slippage': slippage,
        'round_trip_cost': round_trip_cost
    }


def calc_ar_ir_from_scores(
    score_df: pd.DataFrame,
    top_k: int = 30,
    trading_days_per_year: int = 252,
    commission: float = 0.001,
    stamp_duty: float = 0.001,
    slippage: float = 0.0015,
) -> Dict:
    """
    从score文件计算AR和IR

    Args:
        score_df: 包含instrument, datetime, score, label的DataFrame
        top_k: 每日选股数量
        trading_days_per_year: 年化交易日数
        commission: 佣金费率 (单边)
        stamp_duty: 印花税费率 (仅卖出)
        slippage: 滑点费率 (双边)

    Returns:
        dict with AR, IR, daily_excess, cum_excess, etc.
    """
    # 构建预测值Series (MultiIndex: datetime, instrument)
    score_df = score_df.copy()
    score_df['datetime'] = pd.to_datetime(score_df['datetime'])

    predictions = score_df.set_index(['datetime', 'instrument'])['score']

    # 构建label DataFrame
    labels = score_df.set_index(['datetime', 'instrument'])['label']
    label_df = labels.unstack(level='instrument')
    label_df = label_df.sort_index()

    # 调用统一的计算函数
    return calc_ar_ir(
        predictions, label_df, top_k, trading_days_per_year,
        commission, stamp_duty, slippage
    )


def calc_multiseed_ar_ir(
    score_files: List[Union[str, Path]],
    top_k: int = 30,
    universe: Optional[str] = None,
    commission: float = 0.0005,
    stamp_duty: float = 0.001,
    slippage: float = 0.0005,
) -> Dict:
    """
    计算多seed的AR/IR指标

    Args:
        score_files: scores CSV文件路径列表
        top_k: 每日选股数量
        universe: 股票池名称
        commission: 佣金费率 (单边)
        stamp_duty: 印花税费率 (仅卖出)
        slippage: 滑点费率 (双边)

    Returns:
        包含各seed结果和聚合统计的字典
    """
    results = []
    for file_path in score_files:
        print(f"Processing {file_path}...")
        score_df = pd.read_csv(file_path)

        result = calc_ar_ir_from_scores(
            score_df=score_df, top_k=top_k,
            commission=commission, stamp_duty=stamp_duty, slippage=slippage
        )
        result['file'] = str(file_path)
        results.append(result)

    # 聚合统计
    ar_list = [r['AR'] for r in results]
    ir_list = [r['IR'] for r in results]
    ar_net_list = [r['AR_net'] for r in results]
    ir_net_list = [r['IR_net'] for r in results]
    avg_turnover_list = [r['avg_turnover'] for r in results]
    annualized_cost_list = [r['annualized_trading_cost'] for r in results]

    summary = {
        'universe': universe,
        'n_seeds': len(results),
        'top_k': top_k,
        'commission': commission,
        'stamp_duty': stamp_duty,
        'slippage': slippage,
        'round_trip_cost': 2 * (commission + slippage) + stamp_duty,
        'AR': {
            'mean': np.mean(ar_list),
            'std': np.std(ar_list),
            'min': np.min(ar_list),
            'max': np.max(ar_list),
            'values': ar_list
        },
        'IR': {
            'mean': np.mean(ir_list),
            'std': np.std(ir_list),
            'min': np.min(ir_list),
            'max': np.max(ir_list),
            'values': ir_list
        },
        'AR_net': {
            'mean': np.mean(ar_net_list),
            'std': np.std(ar_net_list),
            'min': np.min(ar_net_list),
            'max': np.max(ar_net_list),
            'values': ar_net_list
        },
        'IR_net': {
            'mean': np.mean(ir_net_list),
            'std': np.std(ir_net_list),
            'min': np.min(ir_net_list),
            'max': np.max(ir_net_list),
            'values': ir_net_list
        },
        'avg_turnover': {
            'mean': np.mean(avg_turnover_list),
            'std': np.std(avg_turnover_list)
        },
        'annualized_trading_cost': {
            'mean': np.mean(annualized_cost_list),
            'std': np.std(annualized_cost_list)
        }
    }

    return {
        'summary': summary,
        'seed_results': results
    }


def plot_ar_ir_results(metrics: Dict,
                       figsize: tuple = (14, 10),
                       save_path: Optional[str] = None,
                       plot_net_only: bool = False) -> None:
    """
    绘制AR/IR回测结果图表

    Args:
        metrics: 回测结果字典
        figsize: 图表大小
        save_path: 保存路径
        plot_net_only: 如果为True，只绘制扣除成本后的图表
    """
    results = metrics['seed_results']
    n_seeds = len(results)
    summary = metrics['summary']

    if plot_net_only:
        # 只绘制扣除成本后的3个独立图表
        round_trip_cost = summary.get('round_trip_cost', 0)
        universe = summary.get('universe', 'Unknown')
        top_k = summary.get('top_k', 30)

        # 图1: 扣除成本后累计超额收益曲线
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        for i, result in enumerate(results):
            cum_excess_net = result['cum_excess_net']
            ax1.plot(cum_excess_net.index, cum_excess_net.values,
                    alpha=0.4, linewidth=0.8, label=f'Seed {i}' if i < 5 else '')

        all_cum_excess_net = pd.concat([r['cum_excess_net'] for r in results], axis=1)
        mean_cum_excess_net = all_cum_excess_net.mean(axis=1)
        ax1.plot(mean_cum_excess_net.index, mean_cum_excess_net.values,
                linewidth=2.5, color='red', label='Mean')

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Excess Return')
        ax1.set_title(f"Cumulative Excess Return - Net after Costs ({universe}, top-{top_k}, round-trip: {round_trip_cost*100:.3f}%)")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        if save_path:
            base_path = str(save_path).replace('.png', '_net_cumexcess.png')
            plt.savefig(base_path, dpi=300, bbox_inches='tight')
            print(f"扣除成本后累计收益图已保存: {base_path}")
        plt.close()

        # 图2: 扣除成本后AR
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ar_net_values = summary['AR_net']['values']
        x = np.arange(len(ar_net_values))
        bars = ax2.bar(x, ar_net_values, color='coral', edgecolor='black', alpha=0.8)
        ax2.axhline(summary['AR_net']['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {summary['AR_net']['mean']:.4f}")
        ax2.set_xlabel('Seed')
        ax2.set_ylabel('AR (Annualized Return)')
        ax2.set_title(f"AR by Seed - Net after Costs ({summary['AR_net']['mean']:.4f} ± {summary['AR_net']['std']:.4f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for i, (bar, val) in enumerate(zip(bars, ar_net_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        if save_path:
            base_path = str(save_path).replace('.png', '_net_ar.png')
            plt.savefig(base_path, dpi=300, bbox_inches='tight')
            print(f"扣除成本后AR图已保存: {base_path}")
        plt.close()

        # 图3: 扣除成本后IR
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ir_net_values = summary['IR_net']['values']
        bars = ax3.bar(x, ir_net_values, color='steelblue', edgecolor='black', alpha=0.8)
        ax3.axhline(summary['IR_net']['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {summary['IR_net']['mean']:.4f}")
        ax3.set_xlabel('Seed')
        ax3.set_ylabel('IR (Information Ratio)')
        ax3.set_title(f"IR by Seed - Net after Costs ({summary['IR_net']['mean']:.4f} ± {summary['IR_net']['std']:.4f})")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for i, (bar, val) in enumerate(zip(bars, ir_net_values)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        if save_path:
            base_path = str(save_path).replace('.png', '_net_ir.png')
            plt.savefig(base_path, dpi=300, bbox_inches='tight')
            print(f"扣除成本后IR图已保存: {base_path}")
        plt.close()

        # 图4: 每日换手率曲线
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        for i, result in enumerate(results):
            turnover = result['daily_turnover']
            ax4.plot(turnover.index, turnover.values * 100,
                    alpha=0.4, linewidth=0.8, label=f'Seed {i}' if i < 5 else '')

        all_turnover = pd.concat([r['daily_turnover'] for r in results], axis=1)
        mean_turnover = all_turnover.mean(axis=1) * 100
        ax4.plot(mean_turnover.index, mean_turnover.values,
                linewidth=2.5, color='red', label='Mean')

        ax4.axhline(summary['avg_turnover']['mean'] * 100, color='blue', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f"Avg: {summary['avg_turnover']['mean']*100:.2f}%")
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Daily Turnover (%)')
        ax4.set_title(f"Daily Turnover ({summary['avg_turnover']['mean']*100:.2f}% ± {summary['avg_turnover']['std']*100:.2f}%)")
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        if save_path:
            base_path = str(save_path).replace('.png', '_turnover.png')
            plt.savefig(base_path, dpi=300, bbox_inches='tight')
            print(f"换手率图已保存: {base_path}")
        plt.close()

    else:
        # 原始的综合图表
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # 1. 累计超额收益曲线
        ax1 = fig.add_subplot(gs[0, :])
        for i, result in enumerate(results):
            cum_excess = result['cum_excess']
            ax1.plot(cum_excess.index, cum_excess.values,
                    alpha=0.5, linewidth=0.8, label=f'Seed {i}' if i < 3 else '')

        all_cum_excess = pd.concat([r['cum_excess'] for r in results], axis=1)
        mean_cum_excess = all_cum_excess.mean(axis=1)
        ax1.plot(mean_cum_excess.index, mean_cum_excess.values,
                linewidth=2, color='red', label='Mean')

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('Cumulative Excess Return')
        ax1.set_title(f"Cumulative Excess Return ({summary['universe']}, {n_seeds} seeds, top-{summary['top_k']})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # 2. 日超额收益分布
        ax2 = fig.add_subplot(gs[1, 0])
        all_excess = pd.concat([r['daily_excess'] for r in results])
        ax2.hist(all_excess, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(all_excess.mean(), color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {all_excess.mean():.4f}")
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Daily Excess Return')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f"Daily Excess Distribution (std={all_excess.std():.4f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 组合收益 vs 基准收益
        ax3 = fig.add_subplot(gs[1, 1])
        port_returns = pd.concat([r['daily_portfolio_return'] for r in results])
        bench_returns = pd.concat([r['daily_benchmark_return'] for r in results])
        ax3.hist(port_returns, bins=50, alpha=0.5, color='green', label='Portfolio', edgecolor='black')
        ax3.hist(bench_returns, bins=50, alpha=0.5, color='orange', label='Benchmark', edgecolor='black')
        ax3.axvline(port_returns.mean(), color='green', linestyle='--', linewidth=2)
        ax3.axvline(bench_returns.mean(), color='orange', linestyle='--', linewidth=2)
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.set_title("Portfolio vs Benchmark Returns")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 各seed AR对比
        ax4 = fig.add_subplot(gs[2, 0])
        ar_values = summary['AR']['values']
        ax4.bar(range(len(ar_values)), ar_values, color='steelblue', edgecolor='black')
        ax4.axhline(summary['AR']['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {summary['AR']['mean']:.4f}")
        ax4.set_xlabel('Seed')
        ax4.set_ylabel('AR (Annualized Return)')
        ax4.set_title(f"AR by Seed ({summary['AR']['mean']:.4f} ± {summary['AR']['std']:.4f})")
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. 各seed IR对比
        ax5 = fig.add_subplot(gs[2, 1])
        ir_values = summary['IR']['values']
        ax5.bar(range(len(ir_values)), ir_values, color='coral', edgecolor='black')
        ax5.axhline(summary['IR']['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {summary['IR']['mean']:.4f}")
        ax5.set_xlabel('Seed')
        ax5.set_ylabel('IR (Information Ratio)')
        ax5.set_title(f"IR by Seed ({summary['IR']['mean']:.4f} ± {summary['IR']['std']:.4f})")
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"AR/IR图表已保存: {save_path}")
        else:
            plt.show()

        plt.close()


def save_ar_ir_report(metrics: Dict,
                      output_path: Union[str, Path],
                      format: str = 'excel') -> None:
    """
    保存AR/IR报告
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'excel':
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet1: 汇总统计
            summary_data = []
            summary = metrics['summary']
            summary_data.append({
                '指标': 'AR (年化超额收益)',
                '均值': summary['AR']['mean'],
                '标准差': summary['AR']['std'],
                '最小值': summary['AR']['min'],
                '最大值': summary['AR']['max']
            })
            summary_data.append({
                '指标': 'IR (信息比率)',
                '均值': summary['IR']['mean'],
                '标准差': summary['IR']['std'],
                '最小值': summary['IR']['min'],
                '最大值': summary['IR']['max']
            })
            # 添加扣除成本后的指标
            summary_data.append({
                '指标': 'AR_net (扣除成本后年化超额收益)',
                '均值': summary['AR_net']['mean'],
                '标准差': summary['AR_net']['std'],
                '最小值': summary['AR_net']['min'],
                '最大值': summary['AR_net']['max']
            })
            summary_data.append({
                '指标': 'IR_net (扣除成本后信息比率)',
                '均值': summary['IR_net']['mean'],
                '标准差': summary['IR_net']['std'],
                '最小值': summary['IR_net']['min'],
                '最大值': summary['IR_net']['max']
            })
            summary_data.append({
                '指标': '平均日换手率',
                '均值': summary['avg_turnover']['mean'],
                '标准差': summary['avg_turnover']['std'],
                '最小值': None,
                '最大值': None
            })
            summary_data.append({
                '指标': '年化交易成本',
                '均值': summary['annualized_trading_cost']['mean'],
                '标准差': summary['annualized_trading_cost']['std'],
                '最小值': None,
                '最大值': None
            })
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='汇总统计', index=False)

            # Sheet2: 各seed详情
            seed_data = []
            for i, result in enumerate(metrics['seed_results']):
                seed_data.append({
                    'seed': i,
                    'AR': result['AR'],
                    'IR': result['IR'],
                    'AR_net': result['AR_net'],
                    'IR_net': result['IR_net'],
                    '平均日换手率': result['avg_turnover'],
                    '年化交易成本': result['annualized_trading_cost'],
                    '年化收益': result['annualized_return'],
                    '年化波动': result['annualized_vol'],
                    '夏普比率': result['sharpe_ratio'],
                    '最大回撤': result['max_drawdown'],
                    '日胜率': result['win_rate'],
                    '扣除成本后日胜率': result['win_rate_net'],
                    '交易日数': result['n_days']
                })
            pd.DataFrame(seed_data).to_excel(writer, sheet_name='各seed详情', index=False)

            # Sheet3: 累计超额收益
            cum_excess_data = {}
            for i, result in enumerate(metrics['seed_results']):
                cum_excess_data[f'seed_{i}'] = result['cum_excess']
            cum_excess_df = pd.DataFrame(cum_excess_data)
            cum_excess_df['mean'] = cum_excess_df.mean(axis=1)
            cum_excess_df.to_excel(writer, sheet_name='累计超额收益')

            # Sheet4: 日超额收益
            daily_excess_data = {}
            for i, result in enumerate(metrics['seed_results']):
                daily_excess_data[f'seed_{i}'] = result['daily_excess']
            daily_excess_df = pd.DataFrame(daily_excess_data)
            daily_excess_df['mean'] = daily_excess_df.mean(axis=1)
            daily_excess_df.to_excel(writer, sheet_name='日超额收益')

            # Sheet5: 扣除成本后日超额收益
            daily_excess_net_data = {}
            for i, result in enumerate(metrics['seed_results']):
                daily_excess_net_data[f'seed_{i}'] = result['daily_excess_net']
            daily_excess_net_df = pd.DataFrame(daily_excess_net_data)
            daily_excess_net_df['mean'] = daily_excess_net_df.mean(axis=1)
            daily_excess_net_df.to_excel(writer, sheet_name='扣除成本后日超额收益')

            # Sheet6: 交易成本明细
            trading_cost_data = {}
            for i, result in enumerate(metrics['seed_results']):
                trading_cost_data[f'seed_{i}_turnover'] = result['daily_turnover']
                trading_cost_data[f'seed_{i}_cost'] = result['daily_trading_cost']
            trading_cost_df = pd.DataFrame(trading_cost_data)
            trading_cost_df.to_excel(writer, sheet_name='交易成本明细')

            # Sheet7: 成本参数
            cost_params = {
                '参数': ['佣金(commission)', '印花税(stamp_duty)', '滑点(slippage)', '单边成本', '往返成本'],
                '值': [
                    summary.get('commission', 0.0005),
                    summary.get('stamp_duty', 0.001),
                    summary.get('slippage', 0.0005),
                    summary.get('commission', 0.0005) + summary.get('slippage', 0.0005),
                    summary.get('round_trip_cost', 0.003)
                ],
                '说明': [
                    '单边买卖佣金',
                    '仅卖出时收取',
                    '双边买卖滑点',
                    'commission + slippage',
                    '2*单边成本 + stamp_duty'
                ]
            }
            pd.DataFrame(cost_params).to_excel(writer, sheet_name='成本参数', index=False)

    print(f"AR/IR报告已保存到: {output_path}")


def analyze_ar_ir(
    score_dir: Union[str, Path],
    universe: str,
    n_seeds: int = 5,
    top_k: int = 30,
    output_dir: Optional[str] = None,
    commission: float = 0.0005,
    stamp_duty: float = 0.001,
    slippage: float = 0.0005,
) -> Dict:
    """
    便捷函数：分析某个股票池的AR/IR

    Args:
        score_dir: scores目录路径
        universe: 股票池名称
        n_seeds: seed数量
        top_k: 选股数量
        output_dir: 输出目录
        commission: 佣金费率 (单边), default 0.05%
        stamp_duty: 印花税费率 (仅卖出), default 0.10%
        slippage: 滑点费率 (双边), default 0.05%
    """
    score_dir = Path(score_dir)

    # 查找所有seed文件
    score_files = []
    for i in range(n_seeds):
        file_path = score_dir / f"{universe}_{i}.csv"
        if file_path.exists():
            score_files.append(file_path)
        else:
            print(f"Warning: {file_path} not found")

    if len(score_files) == 0:
        raise FileNotFoundError(f"No score files found for {universe}")

    print(f"Analyzing {len(score_files)} seeds for {universe} (top-{top_k})...")

    # 计算AR/IR
    metrics = calc_multiseed_ar_ir(
        score_files=score_files,
        top_k=top_k,
        universe=universe,
        commission=commission,
        stamp_duty=stamp_duty,
        slippage=slippage
    )

    # 打印摘要
    print("\n" + "="*65)
    print(f"AR/IR Analysis Summary - {universe} (top-{top_k})")
    print("="*65)
    summary = metrics['summary']
    print(f"Seeds: {summary['n_seeds']}, Trading Days: {metrics['seed_results'][0]['n_days']}")
    print(f"\n--- 未扣除成本 ---")
    print(f"AR:  {summary['AR']['mean']:.4f} ± {summary['AR']['std']:.4f}")
    print(f"IR:  {summary['IR']['mean']:.4f} ± {summary['IR']['std']:.4f}")
    print(f"\n--- 扣除交易成本后 ---")
    print(f"佣金: {commission*100:.3f}%, 印花税: {stamp_duty*100:.3f}%, 滑点: {slippage*100:.3f}%")
    print(f"往返成本: {summary['round_trip_cost']*100:.3f}%")
    print(f"平均日换手率: {summary['avg_turnover']['mean']:.2%}")
    print(f"年化交易成本: {summary['annualized_trading_cost']['mean']:.4f}")
    print(f"AR_net: {summary['AR_net']['mean']:.4f} ± {summary['AR_net']['std']:.4f}")
    print(f"IR_net: {summary['IR_net']['mean']:.4f} ± {summary['IR_net']['std']:.4f}")
    print("="*65)

    # 保存报告
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Excel报告
        excel_path = output_dir / f"{universe}_ar_ir_report.xlsx"
        save_ar_ir_report(metrics, excel_path, format='excel')

        # 图表 - 综合图表（包含gross和net对比）
        plot_path = output_dir / f"{universe}_ar_ir_charts.png"
        plot_ar_ir_results(metrics, save_path=plot_path, plot_net_only=False)

        # 图表 - 扣除成本后的独立图表
        plot_ar_ir_results(metrics, save_path=str(plot_path), plot_net_only=True)

        # 保存CSV格式的汇总
        summary_df = pd.DataFrame({
            'seed': list(range(len(summary['AR']['values']))),
            'AR': summary['AR']['values'],
            'IR': summary['IR']['values'],
            'AR_net': summary['AR_net']['values'],
            'IR_net': summary['IR_net']['values']
        })
        summary_csv_path = output_dir / f"{universe}_ar_ir_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nCSV summary saved to: {summary_csv_path}")

        print(f"\n完整报告已生成在: {output_dir}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算AR和IR回测指标')
    parser.add_argument('--universe', type=str, default='csi300', help='股票池名称')
    parser.add_argument('--score_dir', type=str, default='scores', help='scores目录')
    parser.add_argument('--n_seeds', type=int, default=5, help='seed数量')
    parser.add_argument('--top_k', type=int, default=30, help='每日选股数量')
    parser.add_argument('--output_dir', type=str, default='reports', help='输出目录')
    parser.add_argument('--commission', type=float, default=0.0005, help='佣金费率 (单边), default 0.05%%')
    parser.add_argument('--stamp-duty', type=float, default=0.001, dest='stamp_duty', help='印花税费率 (仅卖出), default 0.10%%')
    parser.add_argument('--slippage', type=float, default=0.0005, help='滑点费率 (双边), default 0.05%%')
    parser.add_argument('--no-cost', action='store_true', dest='no_cost', help='禁用交易成本（与旧版本一致）')

    args = parser.parse_args()

    # 如果禁用成本，设置所有成本为0
    if args.no_cost:
        commission = 0.0
        stamp_duty = 0.0
        slippage = 0.0
    else:
        commission = args.commission
        stamp_duty = args.stamp_duty
        slippage = args.slippage

    # 运行分析
    metrics = analyze_ar_ir(
        score_dir=args.score_dir,
        universe=args.universe,
        n_seeds=args.n_seeds,
        top_k=args.top_k,
        output_dir=args.output_dir,
        commission=commission,
        stamp_duty=stamp_duty,
        slippage=slippage
    )
