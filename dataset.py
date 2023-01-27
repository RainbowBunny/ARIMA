import torch
from torch.utils.data import Dataset, DataLoader

def moving_average(data, time_range = 12):
    output = []
    for i in range(len(data) - time_range):
        output.append(torch.mean(data[i : i + time_range]))
    
    return torch.tensor(output)

class TrainingData(Dataset):
    def __init__(self, data, indices, input_length, output_length, time_range = 12):
        self.data = [moving_average(station_data) for station_data in data]
        self.list_IDs = indices
        self.input_length = input_length - time_range + 1
        self.output_length = output_length
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
        index = self.list_IDs[idx]

        for station in self.data:
            station_len = len(station) - self.input_length - self.output_length

            if index < station_len:
                return station[index : index + self.input_length], station[index + self.input_length : index + self.input_length + self.output_length]
            
            index -= station_len
        
        raise ValueError('Index out of range')


class RainFallData(Dataset):
    def __init__(self, data, indices, input_length, output_length):
        self.data = data
        self.list_IDs = indices
        self.input_length = input_length
        self.output_length = output_length
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
        index = self.list_IDs[idx]

        for station in self.data:
            station_len = len(station) - self.input_length - self.output_length

            if index < station_len:
                return station[index : index + self.input_length], station[index + self.input_length : index + self.input_length + self.output_length]
            
            index -= station_len
        
        raise ValueError('Index out of range')

if __name__ == '__main__':
    data = [torch.load(f'data/sample_{0}.pt')]
    sample_dt = RainFallData(data, list(range(3000)), 12, 12)
    sample_train_dt = TrainingData(data, list(range(3000)), 12, 12)
    
    inputs_dt, targets_dt = sample_dt[5]
    inputs_tdt, targets_tdt = sample_train_dt[5]
    
    print(inputs_dt, targets_dt)
    print(moving_average(torch.cat([inputs_dt, targets_dt])))
    print(inputs_tdt, targets_tdt)