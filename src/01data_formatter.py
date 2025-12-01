import pandas as pd
import sys
import glob
from tqdm import tqdm

# Specify the paths to the raw data from a single day.
PATH = "../data/raw_data"

# ========== 经纬度偏移纠正配置 ==========
# 手动设置偏移值来纠正经纬度的小幅线性偏移
LON_OFFSET = -0.00009  # 经度偏移值（度）
LAT_OFFSET = 0.000032  # 纬度偏移值（度）
# ======================================

from tempfile import TemporaryFile


def list_to_df(temp):
    lat = []
    lon = []
    a_x = []
    a_y = []
    v = []
    frame = []
    for i in range(
        3, len(temp) - 6, 6
    ):  # start range from 3 for sample, and 7 for complete. Check this to make it consistent with the format of raw data
        # 应用偏移纠正
        lon.append(float(temp[i + 2]) + LON_OFFSET)  # check lat
        lat.append(float(temp[i + 1]) + LAT_OFFSET)  # check lon
        v.append(temp[i + 3])
        a_x.append(temp[i + 4])
        a_y.append(temp[i + 5])
        frame.append(temp[i + 6])
    df_meta = pd.DataFrame(
        {"id": [temp[0]], "type": [temp[1]], "dist": [temp[2]], "avg_speed": [temp[3]]}
    )
    df_trajectory = pd.DataFrame(
        {
            "id": temp[0],
            "frame": frame,
            "lon": lon,
            "lat": lat,
            "v": v,
            "a_x": a_x,
            "a_y": a_y,
        }
    )
    return df_meta, df_trajectory


def write_list(file):
    with open(file) as f, TemporaryFile("w+") as t:
        next(f)
        list_meta = []
        list_trajectory = []
        for line in tqdm(f):
            h, ln = line, len(line.split("; "))
            temp = h.strip().split("; ")
            m, t = list_to_df(temp)
            list_meta.append(m)
            list_trajectory.append(t)
    return list_meta, list_trajectory


if __name__ == "__main__":
    paths = glob.glob(PATH + "/*.csv")
    print(paths)
    for file in paths:
        print(file)
        filename = file.split("_")[-2]
        # filenames will be the data collection hours e.g. 0900-0930
        print(filename)
        list_meta, list_trajectory = write_list(file)
        # save meta file
        pd.concat(list_meta).to_csv("../data/ok_data/meta_" + filename + ".csv", index=None)

        # save trajectory data
        pd.concat(list_trajectory).to_csv(
            "../data/ok_data/" + filename + ".csv", index=None
        )
