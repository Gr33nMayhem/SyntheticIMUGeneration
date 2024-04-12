# import the necessary packages
import os
import sys

sys.path.append(os.path.join(".."))

import pandas as pd
import torch
import torch.nn as nn
from model_train.preprocess_mag import DataPreprocessMag
from torch.utils.data import DataLoader

from dataloaders.VTT_dataloader import DataSet_VTT
from models.EncoderDecoderModel import EncoderDecoder

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

# reference_point = 'shoulder'
# window_size = 100
# step_size = 5
# imu_position = 'hand'
# learning_rate = 0.0001
# batch_size = 32
# num_epochs = 50
# loss_function = 'mse'

pd.options.display.float_format = '{:.2f}'.format

# check if cuda is available
print(torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

data_path = '../data/VTT_ConIot_Dataset'
IMU_path = data_path + '/IMU'
Keypoint_path = data_path + '/Keypoint'

num_input_features = 18
num_output_features = 1

# main method

preprocess = DataPreprocessMag(reference_point, window_size, step_size)
# get the processed data
keypoint_data, imu_data, sliding_windows = preprocess.processed_data()

# replace any nan value as the mean of the value before and after it
keypoint_data = keypoint_data.fillna(keypoint_data.mean())
imu_data = imu_data.fillna(imu_data.mean())

dataset_train = DataSet_VTT(keypoint_data, imu_data, sliding_windows, 1, flag='train')

train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = (window_size, num_input_features)
output_size = (window_size, num_output_features)

# check if model already exists at save path
print(f'../data/results/encoder_decoder_model_new_{reference_point}_{window_size}_{step_size}_{imu_position}_{learning_rate}_{batch_size}_{num_epochs}_{loss_function}.pth')
if os.path.exists(f'../data/results/encoder_decoder_model_new_{reference_point}_{window_size}_{step_size}_{imu_position}_{learning_rate}_{batch_size}_{num_epochs}_{loss_function}.pth'):
    print('Model already exists at save path. Exiting...')
    # load the model
    model_path = f'../data/results/encoder_decoder_model_new_{reference_point}_{window_size}_{step_size}_{imu_position}_{learning_rate}_{batch_size}_{num_epochs}_{loss_function}.pth'
    model = EncoderDecoder(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
else:
    model = EncoderDecoder(input_size, output_size)

model = model.double().cuda()
if loss_function == 'cross_entropy':
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
else:
    criterion = nn.MSELoss(reduction="mean").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    for i, data in enumerate(train_loader):
        keypoint, imu = data
        keypoint = keypoint.double().to(device)
        # check if keypoint has any nan values
        # print(torch.isnan(keypoint).any())
        imu = imu.double().to(device)
        output = model(keypoint)
        output = output.squeeze(1)
        loss = criterion(output, imu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

# save the model
# create results directory if it does not exist
if not os.path.exists('../data/results'):
    os.makedirs('../data/results')

# save the model with the parameters
torch.save(model.state_dict(),
           f'../data/results/encoder_decoder_model_new_{reference_point}_{window_size}_{step_size}_{imu_position}_{learning_rate}_{batch_size}_{num_epochs}_{loss_function}.pth')
