import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

"""
13compare_lane_stats.py

功能：
- 对比完整的 lane_node_stats（视为“真实值”）与 12test_data.py 生成的 OUTPUT_CSV（视为“模型结果”）；
- 只比较两个文件中都存在的行（按 node_id + time 连接）；
- 对每个车道段（node_id）和每个指标列计算：
  1）相对技能评分 Relative Skill Score (RSS)： 1 - (MAE_model / MAE_baseline)
      - baseline 取该车道该指标在时间序列上的中值；
  2）方向一致性 Directional Accuracy (DA)：
      - 计算时间序列一阶差分的符号是否一致的比例。

使用方式：
1. 在下方“用户可修改参数区域”中，修改输入 / 输出路径以及要比较的指标列；
2. 在命令行中直接运行：
   python src/13compare_lane_stats.py
3. 运行后会在控制台打印前几行结果，并可选择输出到 CSV。
"""

# =================== 用户可修改参数区域 ===================

# 真值的CSV 路径
# LANE_NODE_STATS_PATH = r"../data/draw/fintuned/k0127085203_0001_lane_node_state.csv"
LANE_NODE_STATS_PATH = r'../data/draw/d210191000/d210291000_lane_node_stats.csv'


# 测试数据（模型结果）CSV 路径
OUTPUT_CSV_PATH = r"../data/draw/d210191000/melt/0131绝对时间效果最好/inference_results.csv"

# 可选：要参与比较的指标列列表；
# 如果为 None，则自动从两个文件的公共数值型列中推断（排除 node_id, time）。
VALUE_COLUMNS: Optional[List[str]] = None

# 可选：结果指标输出路径；如果为 None，则只在屏幕上打印，不另存文件。
REPORT_PATH: Optional[str] = r"../data/draw/d210191000/melt/0131绝对时间效果最好/all_metrics.csv"

# =====================================================

KEY_COLUMNS = ["node_id", "time"]


def _infer_value_columns(
    reference_df: pd.DataFrame, comparison_df: pd.DataFrame, user_cols: Optional[List[str]]
) -> List[str]:
    """
    根据用户指定或自动推断需要比较的指标列。
    """
    if user_cols:
        missing = [col for col in user_cols if col not in reference_df.columns or col not in comparison_df.columns]
        if missing:
            raise ValueError(f"❌ 以下列在两个输入文件中不同时存在，无法比较: {missing}")
        return user_cols

    # 自动推断：参考表中除关键键之外的数值型列，且在对比表中也存在
    candidate_cols = [c for c in reference_df.columns if c not in KEY_COLUMNS]
    numeric_cols = [
        c
        for c in candidate_cols
        if pd.api.types.is_numeric_dtype(reference_df[c]) and c in comparison_df.columns
    ]

    if not numeric_cols:
        raise ValueError("❌ 无法自动推断需要比较的数值型列，请手动在 VALUE_COLUMNS 中指定。")
    return numeric_cols


def _prepare_dataframe(path: str) -> pd.DataFrame:
    """
    读取 CSV，并保证包含 node_id / time 两个关键列，且类型一致。
    """
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"❌ 文件 {path} 缺少必要列: {missing}")
    df["node_id"] = df["node_id"].astype(int)
    df["time"] = df["time"].astype(float)
    return df


def _compute_lane_metrics(lane_df: pd.DataFrame, value_cols: List[str]) -> List[Dict[str, float]]:
    """
    对单个 node_id 的数据，按指定指标列计算 RSS 和 DA 等指标。
    """
    results: List[Dict[str, float]] = []
    lane_df = lane_df.sort_values("time")

    for col in value_cols:
        truth = lane_df[f"{col}_truth"].astype(float)
        model = lane_df[f"{col}_model"].astype(float)

        # 如果该列全部缺失，则跳过
        if truth.isna().all() or model.isna().all():
            continue

        # MAE_model
        mae_model = (model - truth).abs().mean()
        # baseline：真实值时间序列中值
        baseline_value = truth.median()
        mae_baseline = (truth - baseline_value).abs().mean()

        # RMSE_model
        rmse_model = np.sqrt(((model - truth) ** 2).mean())
        # RMSE_baseline
        rmse_baseline = np.sqrt(((truth - baseline_value) ** 2).mean())

        # MAPE_model (Mean Absolute Percentage Error)
        # 只在真值非零时计算，避免除零
        mask_nonzero = truth != 0
        if mask_nonzero.sum() > 0:
            mape_model = ((model[mask_nonzero] - truth[mask_nonzero]).abs() / truth[mask_nonzero].abs()).mean() * 100
            mape_baseline = ((truth[mask_nonzero] - baseline_value).abs() / truth[mask_nonzero].abs()).mean() * 100
        else:
            mape_model = np.nan
            mape_baseline = np.nan

        # 相对技能评分 RSS (基于 MAE)
        if np.isclose(mae_baseline, 0.0):
            # 基线误差几乎为 0：如果模型误差也几乎为 0，则认为 RSS = 1，否则无法定义（NaN）
            rss = 1.0 if np.isclose(mae_model, 0.0) else np.nan
        else:
            rss = 1 - (mae_model / mae_baseline)

        # 方向一致性 DA：一阶差分符号匹配率
        truth_diff = truth.diff().dropna().to_numpy()
        model_diff = model.diff().dropna().to_numpy()

        if len(truth_diff) == 0:
            da = np.nan
        else:
            sign_truth = np.sign(truth_diff)
            sign_model = np.sign(model_diff[: len(sign_truth)])
            # 防止长度轻微不一致，统一截断到最小长度
            min_len = min(len(sign_truth), len(sign_model))
            if min_len == 0:
                da = np.nan
            else:
                matches = (sign_truth[:min_len] == sign_model[:min_len]).sum()
                da = matches / min_len

        results.append(
            {
                "node_id": int(lane_df["node_id"].iloc[0]),
                "metric": col,
                "rss": rss,
                "da": da,
                "mae_model": mae_model,
                "mae_baseline": mae_baseline,
                "rmse_model": rmse_model,
                "rmse_baseline": rmse_baseline,
                "mape_model": mape_model,
                "mape_baseline": mape_baseline,
                "n_points": len(truth),
                "n_deltas": max(len(truth) - 1, 0),
            }
        )
    return results


def compare_lane_stats(
    lane_node_stats_path: str,
    output_csv_path: str,
    value_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    主比较函数：
    - 读取两个 CSV，按 node_id + time 做内连接（只保留两个文件都存在的行）；
    - 对每个 node_id、每个指标列计算 RSS 和 DA。
    """
    print("📦 正在读取完整 lane_node_stats（真实值）...")
    reference_df = _prepare_dataframe(lane_node_stats_path)
    print(f"✅ 真实值记录数: {len(reference_df)}")

    print("📦 正在读取测试数据 OUTPUT_CSV（模型结果）...")
    comparison_df = _prepare_dataframe(output_csv_path)
    print(f"✅ 模型结果记录数: {len(comparison_df)}")

    value_cols = _infer_value_columns(reference_df, comparison_df, value_columns)
    print(f"📊 将参与比较的指标列: {value_cols}")

    print("🔄 正在按 node_id + time 对齐两个数据集（只保留重叠行）...")
    merged = reference_df.merge(
        comparison_df,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_truth", "_model"),
    )

    if merged.empty:
        raise ValueError("❌ 两个文件在 (node_id, time) 上没有重叠行，无法比较。")

    print(f"✅ 重叠行数: {len(merged)}，涉及车道数: {merged['node_id'].nunique()}")

    metrics: List[Dict[str, float]] = []
    for node_id, lane_group in merged.groupby("node_id"):
        lane_results = _compute_lane_metrics(lane_group, value_cols)
        metrics.extend(lane_results)

    result_df = pd.DataFrame(metrics)
    
    # 确保数值列为正确的数据类型，避免 CSV 中数值前出现单引号
    numeric_cols = ["rss", "da", "mae_model", "mae_baseline", "rmse_model", "rmse_baseline", 
                    "mape_model", "mape_baseline", "n_points", "n_deltas"]
    for col in numeric_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
    
    # 对浮点数值列保留4位小数（整数列保持不变）
    float_cols = ["rss", "da", "mae_model", "mae_baseline", "rmse_model", "rmse_baseline", 
                  "mape_model", "mape_baseline"]
    for col in float_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(4)
    
    return result_df


if __name__ == "__main__":
    # 运行入口：直接使用上方用户可修改的全局变量
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    metrics_df = compare_lane_stats(LANE_NODE_STATS_PATH, OUTPUT_CSV_PATH, VALUE_COLUMNS)

    if REPORT_PATH:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        # 使用 na_rep='' 将 NaN 保存为空字符串，避免 Excel 将数值识别为文本
        metrics_df.to_csv(REPORT_PATH, index=False, encoding="utf-8", na_rep="")
        print(f"💾 全部指标结果已保存到: {REPORT_PATH}")

