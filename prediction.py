import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from dataset import RainFallData, moving_average

def arg_def(default_checkpoint_dir = 'checkpoints'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type = str, default = default_checkpoint_dir)
    parser.add_argument('--batch_size', type = int, default = 64)

    args = parser.parse_args()
    args.log_interval = 30
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.saved_checkpoint = os.path.join(args.checkpoint_dir, "model_checkpoint.pth")

    return args

def show_prediction(model, data, args):
    inputs, trues = data
    potato = inputs
    print(inputs.shape)
    inputs = moving_average(inputs).to(args.device)
    print(inputs.shape)

    current_avg = inputs[-1]

    outputs = model(inputs)

    predictions = []
    for i in range(12):
        predictions.append(outputs[i].detach() * 12 - current_avg * 12 + potato[-12 + i])
        current_avg = outputs[i].detach()
    
    plt.plot(range(48), potato, label = 'Input data')
    plt.plot(range(48, 60), np.array(predictions), label = 'Predictions')
    plt.plot(range(48, 60), trues, label = 'Truth')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    args = arg_def()

    input_length = 48
    output_length = 12
    time_range = 12

    data = [torch.load(f'data/sample_{i}.pt') for i in range(11)]

    pred_data = RainFallData(data, list(range(3000)),input_length, output_length)

    loaded_checkpoint = torch.load(args.saved_checkpoint, torch.device(args.device))
    model = Model(input_length, output_length, time_range)
    model.load_state_dict(loaded_checkpoint['best_model'])
    model.to(args.device)

    show_prediction(model, pred_data[30], args)