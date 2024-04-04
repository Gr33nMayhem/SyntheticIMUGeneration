# import the necessary packages
import os
import sys
import logging

sys.path.append(os.path.join(".."))

import pandas as pd
import torch
import torch.nn as nn
from model_train.preprocess import DataPreprocess
from torch.utils.data import DataLoader

from dataloaders.VTT_dataloader import DataSet_VTT
from models.EncoderDecoderModel import EncoderDecoder
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='Train Encoder Decoder Model')
parser.add_argument('--reference_point', type=str, default='shoulder',
                    help='Reference point for getting relative keypoints')
parser.add_argument('--window_size', type=int, default=25,
                    help='Sliding window size')
parser.add_argument('--step_size', type=int, default=5,
                    help='Sliding window step size')
parser.add_argument('--imu_position', type=str, default='hand',
                    help='IMU position to use for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, )
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=150,
                    help='Number of epochs for training')
parser.add_argument('--loss_function', type=str, default='cross_entropy', )
args = parser.parse_args()
reference_point = args.reference_point
window_size = args.window_size
step_size = args.step_size
imu_position = args.imu_position
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
loss_function = args.loss_function

pd.options.display.float_format = '{:.2f}'.format

# check if cuda is available
logging.info(f"Cuda is available: {torch.cuda.is_available()}")
torch.autograd.set_detect_anomaly(True)

data_path = '../data/VTT_ConIot_Dataset'
IMU_path = data_path + '/IMU'
Keypoint_path = data_path + '/Keypoint'
activities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
users = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
num_input_features = 36
num_output_features = 3

# main method

preprocess = DataPreprocess(reference_point, window_size, step_size, imu_position)
# get the processed data
keypoint_data, imu_data, sliding_windows = preprocess.processed_data()

# replace any nan value as the mean of the value before and after it
keypoint_data = keypoint_data.fillna(keypoint_data.mean())
imu_data = imu_data.fillna(imu_data.mean())

dataset_test = DataSet_VTT(keypoint_data, imu_data, sliding_windows, 1, flag='test')

test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = (window_size, num_input_features)
output_size = (window_size, num_output_features)
model = EncoderDecoder(input_size, output_size)
model = model.double().cuda()

# load the model
model_path = f'../data/results/encoder_decoder_model_{reference_point}_{window_size}_{step_size}_{imu_position}_{learning_rate}_{batch_size}_{num_epochs}_{loss_function}.pth'
# if model exists load it
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # use the test_loader to get the test accuracy
    test_loss = 0
    count = 0
    logging.info('Model loaded')
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            keypoint, imu = data
            keypoint = keypoint.double().to(device)
            imu = imu.double().to(device)
            output = model(keypoint)
            output = output.squeeze(1)
            # calculate the loss
            loss = nn.MSELoss(reduction="mean")(output, imu)
            test_loss += loss.item()
            if count % 10 == 0:
                output = output.cpu().detach().numpy()
                imu = imu.cpu().detach().numpy()
                # from ouput and imu get the first sample
                output = output[0]
                imu = imu[0]
                plt.figure()
                # make plot between -3 and 3
                plt.ylim(-3, 3)
                plt.plot(output, label=['Output X', 'Output Y', 'Output Z'], linestyle='--')
                plt.plot(imu, label=['IMU X', 'IMU Y', 'IMU Z'], linestyle='-')
                plt.legend()
                plt.show()
    logging.info(f'Test Loss: {test_loss / len(test_loader)}')
else:
    logging.warning('Error: Model not found')
