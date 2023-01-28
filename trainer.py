import torch
import numpy as np
import logging
from datetime import datetime
import argparse
import os

from torch.utils.data import Dataset, DataLoader

from dataset import TrainingData, RainFallData
from model import Model

def run_train(train_ds, model, optimizer, loss_func, args, coef = None):
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, num_workers = 4)
    train_mse = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)
        loss = loss_func(outputs, targets)

        train_mse.append(loss.item() / targets.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == args.log_interval - 1:
            logging.info(f'    Batch {batch_idx + 1}: MSE = {train_mse[-1]}')
    
    return round(np.sqrt(np.mean(train_mse)), 5)

def run_eval(valid_ds, model, loss_function, args):
    valid_loader = DataLoader(valid_ds, batch_size = args.batch_size, shuffle = False, num_workers = 4)
    valid_mse = []
    preds = []
    trues = []

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            preds.append(outputs)
            trues.append(targets)

            valid_mse.append(loss.item() / targets.shape[0])
    
    valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)

    return valid_mse, preds, trues

def arg_def(default_checkpoint_dir = 'checkpoints'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type = str, default = default_checkpoint_dir)
    parser.add_argument('--batch_size', type = int, default = 64)

    args = parser.parse_args()
    args.log_interval = 30
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.saved_checkpoint = os.path.join(args.checkpoint_dir, "model_checkpoint.pth")

    return args

# Set up the hyperparameters
LEARNING_RATE = 2e-5
n_epochs = 4000

if __name__ == '__main__':
    logging_configs = {
        'filename': 'training_log.log',
        'level': logging.INFO, 
    }

    logging.basicConfig(**logging_configs)
    args = arg_def()

    logging.info(args)

    input_length = 48
    output_length = 12
    time_range = 12

    data = [torch.load(f'data/sample_{i}.pt') for i in range(11)]
    train_ds = TrainingData(data, list(range(2000)), input_length, output_length, time_range)
    valid_ds = TrainingData(data, list(range(2000, 3000)), input_length, output_length)

    model = Model(input_length, output_length, time_range).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    best_model = model

    if os.path.exists(args.saved_checkpoint):
        loaded_checkpoint = torch.load(args.saved_checkpoint, torch.device(args.device))
        start_epoch = loaded_checkpoint['epoch'] + 1

        model.load_state_dict(loaded_checkpoint['model'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer'])
        best_model.load_state_dict(loaded_checkpoint['best_model'])

        train_mse = loaded_checkpoint['train_mse']
        valid_mse = loaded_checkpoint['valid_mse']

        min_mse = loaded_checkpoint['min_mse']
    else:
        os.system('mkdir ' + args.checkpoint_dir)
        
        start_epoch = 0
        train_mse = []
        valid_mse = []
        
        min_mse = 10000000000000

    logging.info(f"Started training model at {datetime.now()}")
    for i in range(start_epoch, n_epochs):
        logging.info(f"Started epoch {i} at {datetime.now()}")

        model.train()
        train_mse.append(run_train(train_ds, model, optimizer, torch.nn.MSELoss(), args, 0))
        
        model.eval()
        mse, _, _ = run_eval(valid_ds, model, torch.nn.MSELoss(), args)
        valid_mse.append(mse)

        if valid_mse[-1] < min_mse:
            min_mse = valid_mse[-1]
            best_model = model
        
        torch.save({
            'epoch': i,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_mse': train_mse,
            'valid_mse': valid_mse,
            'min_mse': min_mse,
            'best_model': best_model.state_dict()
        }, args.saved_checkpoint)

        logging.info("Resulted MSE:")
        logging.info(f"Train_mse: {train_mse[-1]}")
        logging.info(f"Validate_mse: {valid_mse[-1]}")


        logging.info(f"Finished epoch {i} at {datetime.now()}")