import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def process(filename, station):
    df = pd.read_excel(filename, sheet_name=station)
    df = pd.DataFrame(df)
    station_info = df.iloc[5 : 35, 31 : 43].dropna()
    return station_info.to_numpy().flatten()

def new_diff(df, idx):
    out = []
    for i in range(len(df) - idx):
        out.append(df[i + idx] - df[i])
    return np.array(out)

def moving_average(df, idx):
    out = []
    for i in range(len(df) - idx):
        out.append(sum(df[i : i + idx]) / idx)
    return out

if __name__ == "__main__":
    file_name = "Rainfall_data.xls"
    file = pd.ExcelFile(file_name)

    xticks = None

    mean_list = []
    std_list = []
    for station in file.sheet_names[1:]:
        rain_fall = process(file_name, station)
        print(station, np.mean(rain_fall), np.std(rain_fall))
        mean_list.append(np.mean(rain_fall))
        std_list.append(np.std(rain_fall))
    
    print(mean_list)
    print(std_list)