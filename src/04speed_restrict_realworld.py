# 为了解决有的速度严重超速没有被剔除的问题

import pandas as pd
import numpy as np

# ✅ 读取 CSV（自动识别空值如 "", " ", 或缺失）
df = pd.read_csv(r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\trajectory_with_laneid_wrong\0127085212_0001.csv", keep_default_na=True, na_values=["", " ", "NULL", "null", "N/A"])

# ✅ 安全处理：只对“速度”列中「数值型且 > 75」的单元格置空；其余（含原始 NaN/空）保持不变
# 方法：先确保转为数值，错误转为 NaN；再用布尔索引赋值
df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")  # 非数字→转为 NaN
mask = df["speed_kmh"].notna() & (df["speed_kmh"] > 75)  # 仅对「有效数字且 > 75」生效
df.loc[mask, "speed_kmh"] = pd.NA  # 置为 pandas 空值（最规范）

# ✅ 保存为 CSV：所有 NA 自动输出为空字段（无引号，符合标准 CSV）
df.to_csv(r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\trajectory_with_laneid\0127085212_0001.csv", index=False, na_rep="")
print("✅ 处理完成！已保存至 output.csv")