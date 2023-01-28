import torch

import pandas as pd

import os

def process(filename, station):
    df = pd.read_excel(filename, sheet_name=station)
    df = pd.DataFrame(df)
    station_info = df.iloc[5 : 36, 31 : 43].dropna()
    tensor = torch.from_numpy(station_info.to_numpy())
    return tensor.flatten()


if __name__ == "__main__":
    file_name = "Rainfall_data.xls"
    file = pd.ExcelFile(file_name)
    print(file.sheet_names)

    if not os.path.exists('data'):
        os.mkdir('data')
    
    stations = []
    for station in file.sheet_names[1:]:
        stations.append(process(file_name, station))
    
    for i in range(len(stations)):
        torch.save(stations[i], f'data/sample_{i}.pt')