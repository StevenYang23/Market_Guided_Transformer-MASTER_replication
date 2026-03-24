"""
量化模型评估指标计算模块
支持IC、RANKIC、ICIR、RANKICIR的计算和可视化
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats

# 可视化
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calc_daily_ic(df: pd.DataFrame, score_col: str = 'score', label_col: str = 'label') -> Dict[str, float]:
    """
    计算单日的IC和RANKIC

    Args:
        df: DataFrame包含score和label列
        score_col: 预测分数列名
        label_col: 真实标签列名

    Returns:
        {'IC': pearson相关系数, 'RANKIC': spearman相关系数}
    """
    if len(df) < 2:
        return {'IC': np.nan, 'RANKIC': np.nan}

    # 去除缺失值
    mask = df[score_col].notna() & df[label_col].notna()
    filtered = df[mask]

    if len(filtered) < 2:
        return {'IC': np.nan, 'RANKIC': np.nan}

    ic = filtered[score_col].corr(filtered[label_col], method='pearson')
    rankic = filtered[score_col].corr(filtered[label_col], method='spearman')

    return {'IC': ic, 'RANKIC': rankic}


def calc_ic_series(df: pd.DataFrame, date_col: str = 'datetime',
                   score_col: str = 'score', label_col: str = 'label') -> pd.DataFrame:
    """
    计算每日IC序列

    Args:
        df: DataFrame包含datetime, score, label列
        date_col: 日期列名
        score_col: 预测分数列名
        label_col: 真实标签列名

    Returns:
        DataFrame (index=date, columns=['IC', 'RANKIC'])
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily_ic = []
    for date, group in df.groupby(date_col):
        ic_dict = calc_daily_ic(group, score_col, label_col)
        ic_dict['date'] = date
        daily_ic.append(ic_dict)

    result = pd.DataFrame(daily_ic)
    result.set_index('date', inplace=True)
    result.sort_index(inplace=True)

    return result


def calc_ic_metrics(ic_series: pd.Series) -> Dict[str, float]:
    """
    从IC序列计算统计指标

    Args:
        ic_series: 每日IC值序列

    Returns:
        包含IC均值、标准差、ICIR、胜率等统计量的字典
    """
    valid_ic = ic_series.dropna()

    if len(valid_ic) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ir': np.nan,
            'win_rate': np.nan,
            'positive_rate': np.nan,
            't_stat': np.nan,
            'p_value': np.nan
        }

    mean_ic = valid_ic.mean()
    std_ic = valid_ic.std()
    ir = mean_ic / std_ic if std_ic > 0 else np.nan

    # 胜率：IC > 0的比例
    positive_rate = (valid_ic > 0).mean()
    # 显著胜率：|IC| > 0.02的比例
    win_rate = (valid_ic.abs() > 0.02).mean()

    # t检验
    t_stat, p_value = stats.ttest_1samp(valid_ic, 0)

    return {
        'mean': mean_ic,
        'std': std_ic,
        'ir': ir,
        'win_rate': win_rate,
        'positive_rate': positive_rate,
        't_stat': t_stat,
        'p_value': p_value,
        'min': valid_ic.min(),
        'max': valid_ic.max(),
        'median': valid_ic.median()
    }


def process_single_seed(file_path: Union[str, Path],
                        score_col: str = 'score',
                        label_col: str = 'label') -> Dict:
    """
    处理单个seed的预测结果

    Args:
        file_path: CSV文件路径
        score_col: 预测分数列名
        label_col: 真实标签列名

    Returns:
        包含IC序列和统计指标的字典
    """
    df = pd.read_csv(file_path)

    # 确保列存在
    if score_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain '{score_col}' and '{label_col}' columns")

    # 计算每日IC
    ic_series = calc_ic_series(df, score_col=score_col, label_col=label_col)

    # 计算统计指标
    ic_metrics = calc_ic_metrics(ic_series['IC'])
    rankic_metrics = calc_ic_metrics(ic_series['RANKIC'])

    return {
        'file': str(file_path),
        'ic_series': ic_series,
        'ic_metrics': ic_metrics,
        'rankic_metrics': rankic_metrics,
        'n_days': len(ic_series)
    }


def calc_multiseed_metrics(
    score_files: List[Union[str, Path]],
    universe: Optional[str] = None,
    score_col: str = 'score',
    label_col: str = 'label'
) -> Dict:
    """
    计算多seed的IC指标

    Args:
        score_files: 多个seed的CSV文件路径列表
        universe: 股票池名称（如'csi300', 'csi800'）
        score_col: 预测分数列名
        label_col: 真实标签列名

    Returns:
        包含各seed结果和聚合统计的字典
    """
    results = []

    for file_path in score_files:
        result = process_single_seed(file_path, score_col, label_col)
        results.append(result)

    # 合并IC序列（按日期对齐）
    ic_dfs = []
    rankic_dfs = []

    for i, result in enumerate(results):
        ic_df = result['ic_series'][['IC']].rename(columns={'IC': f'seed_{i}'})
        rankic_df = result['ic_series'][['RANKIC']].rename(columns={'RANKIC': f'seed_{i}'})
        ic_dfs.append(ic_df)
        rankic_dfs.append(rankic_df)

    # 按日期对齐
    ic_combined = pd.concat(ic_dfs, axis=1)
    rankic_combined = pd.concat(rankic_dfs, axis=1)

    # 计算平均IC序列
    ic_combined['mean'] = ic_combined.mean(axis=1, skipna=True)
    rankic_combined['mean'] = rankic_combined.mean(axis=1, skipna=True)

    # 计算各seed指标的统计量
    ic_means = [r['ic_metrics']['mean'] for r in results]
    ic_stds = [r['ic_metrics']['std'] for r in results]
    ic_irs = [r['ic_metrics']['ir'] for r in results]

    rankic_means = [r['rankic_metrics']['mean'] for r in results]
    rankic_stds = [r['rankic_metrics']['std'] for r in results]
    rankic_irs = [r['rankic_metrics']['ir'] for r in results]

    # seed间相关性
    ic_corr = ic_combined.drop('mean', axis=1).corr()
    rankic_corr = rankic_combined.drop('mean', axis=1).corr()

    summary = {
        'universe': universe,
        'n_seeds': len(results),
        'n_days': results[0]['n_days'],
        'IC': {
            'mean': np.mean(ic_means),
            'std': np.std(ic_means),
            'min': np.min(ic_means),
            'max': np.max(ic_means),
            'ir_mean': np.mean(ic_irs),
            'ir_std': np.std(ic_irs)
        },
        'RANKIC': {
            'mean': np.mean(rankic_means),
            'std': np.std(rankic_stds),
            'min': np.min(rankic_means),
            'max': np.max(rankic_means),
            'ir_mean': np.mean(rankic_irs),
            'ir_std': np.std(rankic_irs)
        },
        'seed_correlation': {
            'ic_mean': ic_corr.values[np.triu_indices_from(ic_corr.values, k=1)].mean(),
            'rankic_mean': rankic_corr.values[np.triu_indices_from(rankic_corr.values, k=1)].mean()
        }
    }

    return {
        'summary': summary,
        'seed_results': results,
        'ic_combined': ic_combined,
        'rankic_combined': rankic_combined,
        'ic_correlation': ic_corr,
        'rankic_correlation': rankic_corr
    }


def save_metrics_report(
    metrics: Dict,
    output_path: Union[str, Path],
    format: str = 'excel'
) -> None:
    """
    保存指标报告到文件

    Args:
        metrics: calc_multiseed_metrics的返回结果
        output_path: 输出文件路径
        format: 'excel' 或 'csv'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'excel':
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet1: 汇总统计
            summary_data = []
            for metric_name in ['IC', 'RANKIC']:
                data = metrics['summary'][metric_name]
                summary_data.append({
                    '指标': metric_name,
                    '均值': f"{data['mean']:.4f}",
                    '标准差': f"{data['std']:.4f}",
                    'IR均值': f"{data['ir_mean']:.4f}",
                    'IR标准差': f"{data['ir_std']:.4f}",
                    '最小值': f"{data['min']:.4f}",
                    '最大值': f"{data['max']:.4f}"
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)

            # Sheet2: 各seed详情
            seed_data = []
            for i, result in enumerate(metrics['seed_results']):
                ic_m = result['ic_metrics']
                ric_m = result['rankic_metrics']
                seed_data.append({
                    'seed': i,
                    'IC': ic_m['mean'],
                    'IC_std': ic_m['std'],
                    'ICIR': ic_m['ir'],
                    'IC胜率': ic_m['positive_rate'],
                    'RANKIC': ric_m['mean'],
                    'RANKIC_std': ric_m['std'],
                    'RANKICIR': ric_m['ir'],
                    'RANKIC胜率': ric_m['positive_rate']
                })
            pd.DataFrame(seed_data).to_excel(writer, sheet_name='各seed详情', index=False)

            # Sheet3: 每日IC
            daily_ic = metrics['ic_combined'].copy()
            daily_ic.columns = [f'IC_{c}' if c != 'mean' else 'IC_mean' for c in daily_ic.columns]
            daily_rankic = metrics['rankic_combined'].copy()
            daily_rankic.columns = [f'RANKIC_{c}' if c != 'mean' else 'RANKIC_mean' for c in daily_rankic.columns]
            daily_combined = pd.concat([daily_ic, daily_rankic], axis=1)
            daily_combined.to_excel(writer, sheet_name='每日IC')

            # Sheet4: Seed间IC相关性
            metrics['ic_correlation'].to_excel(writer, sheet_name='IC相关性')

            # Sheet5: Seed间RANKIC相关性
            metrics['rankic_correlation'].to_excel(writer, sheet_name='RANKIC相关性')

    else:  # CSV格式
        # 保存为多个CSV文件
        base_path = output_path.parent / output_path.stem

        # 汇总统计
        summary_df.to_csv(f"{base_path}_summary.csv", index=False)

        # 每日IC
        metrics['ic_combined'].to_csv(f"{base_path}_daily_ic.csv")

    print(f"报告已保存到: {output_path}")


# ==================== 可视化功能 ====================

def plot_ic_timeseries(metrics: Dict,
                       figsize: Tuple[int, int] = (14, 8),
                       save_path: Optional[str] = None) -> None:
    """
    绘制IC时序图

    Args:
        metrics: calc_multiseed_metrics的返回结果
        figsize: 图像大小
        save_path: 保存路径（None则显示图像）
    """
    ic_combined = metrics['ic_combined']
    rankic_combined = metrics['rankic_combined']

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # IC时序图
    ax1 = axes[0]
    for col in ic_combined.columns:
        if col != 'mean':
            ax1.plot(ic_combined.index, ic_combined[col],
                    alpha=0.3, linewidth=0.8, color='gray', label='_nolegend_')

    ax1.plot(ic_combined.index, ic_combined['mean'],
            linewidth=2, color='blue', label='Mean IC')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.axhline(y=ic_combined['mean'].mean(), color='red', linestyle='--',
               linewidth=1, label=f"Avg: {ic_combined['mean'].mean():.4f}")
    ax1.set_ylabel('IC')
    ax1.set_title(f"IC Time Series ({metrics['summary']['universe']}, "
                 f"{metrics['summary']['n_seeds']} seeds)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # RANKIC时序图
    ax2 = axes[1]
    for col in rankic_combined.columns:
        if col != 'mean':
            ax2.plot(rankic_combined.index, rankic_combined[col],
                    alpha=0.3, linewidth=0.8, color='gray', label='_nolegend_')

    ax2.plot(rankic_combined.index, rankic_combined['mean'],
            linewidth=2, color='green', label='Mean RANKIC')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.axhline(y=rankic_combined['mean'].mean(), color='red', linestyle='--',
               linewidth=1, label=f"Avg: {rankic_combined['mean'].mean():.4f}")
    ax2.set_ylabel('RANKIC')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 格式化x轴日期
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IC时序图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ic_distribution(metrics: Dict,
                         figsize: Tuple[int, int] = (12, 5),
                         save_path: Optional[str] = None) -> None:
    """
    绘制IC分布直方图

    Args:
        metrics: calc_multiseed_metrics的返回结果
        figsize: 图像大小
        save_path: 保存路径
    """
    ic_mean = metrics['ic_combined']['mean']
    rankic_mean = metrics['rankic_combined']['mean']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # IC分布
    ax1 = axes[0]
    ax1.hist(ic_mean, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(ic_mean.mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {ic_mean.mean():.4f}")
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('IC')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f"IC Distribution (std={ic_mean.std():.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # RANKIC分布
    ax2 = axes[1]
    ax2.hist(rankic_mean, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(rankic_mean.mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {rankic_mean.mean():.4f}")
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('RANKIC')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f"RANKIC Distribution (std={rankic_mean.std():.4f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IC分布图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_correlation_heatmap(metrics: Dict,
                             figsize: Tuple[int, int] = (10, 8),
                             save_path: Optional[str] = None) -> None:
    """
    绘制Seed间IC相关性热力图

    Args:
        metrics: calc_multiseed_metrics的返回结果
        figsize: 图像大小
        save_path: 保存路径
    """
    ic_corr = metrics['ic_correlation']
    rankic_corr = metrics['rankic_correlation']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # IC相关性
    ax1 = axes[0]
    sns.heatmap(ic_corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax1,
                square=True, cbar_kws={'shrink': 0.8})
    ax1.set_title(f"IC Correlation Matrix\n"
                 f"(Mean: {metrics['summary']['seed_correlation']['ic_mean']:.3f})")

    # RANKIC相关性
    ax2 = axes[1]
    sns.heatmap(rankic_corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax2,
                square=True, cbar_kws={'shrink': 0.8})
    ax2.set_title(f"RANKIC Correlation Matrix\n"
                 f"(Mean: {metrics['summary']['seed_correlation']['rankic_mean']:.3f})")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相关性热力图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ic_cumsum(metrics: Dict,
                   figsize: Tuple[int, int] = (14, 6),
                   save_path: Optional[str] = None) -> None:
    """
    绘制IC累积和图（反映IC的稳定性）

    Args:
        metrics: calc_multiseed_metrics的返回结果
        figsize: 图像大小
        save_path: 保存路径
    """
    ic_combined = metrics['ic_combined']
    rankic_combined = metrics['rankic_combined']

    fig, ax = plt.subplots(figsize=figsize)

    # 计算累积和
    ic_cumsum = ic_combined['mean'].cumsum()
    rankic_cumsum = rankic_combined['mean'].cumsum()

    ax.plot(ic_cumsum.index, ic_cumsum.values,
           linewidth=2, color='blue', label='IC Cumulative Sum')
    ax.plot(rankic_cumsum.index, rankic_cumsum.values,
           linewidth=2, color='green', label='RANKIC Cumulative Sum')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Sum')
    ax.set_title(f"IC Cumulative Sum ({metrics['summary']['universe']})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IC累积图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_rolling_ic(metrics: Dict,
                   window: int = 20,
                   figsize: Tuple[int, int] = (14, 6),
                   save_path: Optional[str] = None) -> None:
    """
    绘制滚动IC均值图

    Args:
        metrics: calc_multiseed_metrics的返回结果
        window: 滚动窗口大小（默认20天）
        figsize: 图像大小
        save_path: 保存路径
    """
    ic_combined = metrics['ic_combined']
    rankic_combined = metrics['rankic_combined']

    fig, ax = plt.subplots(figsize=figsize)

    # 计算滚动均值
    ic_rolling = ic_combined['mean'].rolling(window=window, min_periods=1).mean()
    rankic_rolling = rankic_combined['mean'].rolling(window=window, min_periods=1).mean()

    ax.plot(ic_rolling.index, ic_rolling.values,
           linewidth=2, color='blue', label=f'IC ({window}d MA)')
    ax.plot(rankic_rolling.index, rankic_rolling.values,
           linewidth=2, color='green', label=f'RANKIC ({window}d MA)')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Mean')
    ax.set_title(f"Rolling IC (window={window} days, {metrics['summary']['universe']})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"滚动IC图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_full_report(metrics: Dict,
                         output_dir: Union[str, Path],
                         name: str = 'ic_report') -> None:
    """
    生成完整的IC分析报告（包含所有图表和Excel）

    Args:
        metrics: calc_multiseed_metrics的返回结果
        output_dir: 输出目录
        name: 报告名称前缀
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存Excel报告
    excel_path = output_dir / f"{name}.xlsx"
    save_metrics_report(metrics, excel_path, format='excel')

    # 生成所有图表
    plot_ic_timeseries(metrics, save_path=output_dir / f"{name}_timeseries.png")
    plot_ic_distribution(metrics, save_path=output_dir / f"{name}_distribution.png")
    plot_correlation_heatmap(metrics, save_path=output_dir / f"{name}_correlation.png")
    plot_ic_cumsum(metrics, save_path=output_dir / f"{name}_cumsum.png")
    plot_rolling_ic(metrics, window=20, save_path=output_dir / f"{name}_rolling20.png")

    print(f"\n完整报告已生成在: {output_dir}")


# ==================== 便捷函数 ====================

def analyze_universe(score_dir: Union[str, Path],
                     universe: str,
                     n_seeds: int = 5,
                     output_dir: Optional[str] = None) -> Dict:
    """
    便捷函数：分析某个股票池的所有seed结果

    Args:
        score_dir: scores目录路径
        universe: 股票池名称 ('csi300' 或 'csi800')
        n_seeds: seed数量
        output_dir: 输出目录（None则不保存）

    Returns:
        指标字典
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

    print(f"Analyzing {len(score_files)} seeds for {universe}...")

    # 计算指标
    metrics = calc_multiseed_metrics(score_files, universe=universe)

    # 打印摘要
    print("\n" + "="*50)
    print(f"IC Analysis Summary - {universe}")
    print("="*50)
    summary = metrics['summary']
    print(f"Seeds: {summary['n_seeds']}, Days: {summary['n_days']}")
    print(f"\nIC:   {summary['IC']['mean']:.4f} ± {summary['IC']['std']:.4f}, "
          f"ICIR: {summary['IC']['ir_mean']:.4f} ± {summary['IC']['ir_std']:.4f}")
    print(f"RANKIC: {summary['RANKIC']['mean']:.4f} ± {summary['RANKIC']['std']:.4f}, "
          f"RANKICIR: {summary['RANKIC']['ir_mean']:.4f} ± {summary['RANKIC']['ir_std']:.4f}")
    print(f"\nSeed Correlation - IC: {summary['seed_correlation']['ic_mean']:.3f}, "
          f"RANKIC: {summary['seed_correlation']['rankic_mean']:.3f}")

    # 保存报告
    if output_dir:
        generate_full_report(metrics, output_dir, name=f"{universe}_report")

    return metrics


if __name__ == "__main__":
    # 示例用法
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Calculate metrics for stock prediction models')
    parser.add_argument('universe', type=str, nargs='?', default='csi300',
                        choices=['csi300', 'csi800'], help='Stock universe to analyze')
    parser.add_argument('--output_dir', type=str, default='reports',
                        help='Output directory for metrics results')
    args = parser.parse_args()

    # 分析指定universe
    metrics = analyze_universe(
        score_dir='scores',
        universe=args.universe,
        n_seeds=5,
        output_dir=args.output_dir
    )
