# import the necessary packages
import os
import sys

sys.path.append(os.path.join(".."))

import torch.nn as nn
import torch
from models.EncoderDecoderModel import EncoderDecoder
from torch.utils.data import DataLoader

from dataloaders.VTT_dataloader import DataSet_VTT
from models.EncoderDecoderModel import EncoderDecoder

import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, participants, input_size, keypoint_data, imu_data, sliding_window, output_size, device,
                 save_path, batch_size=32,
                 num_epochs=50, learning_rate=0.0001, loss_function='mse'):
        self.keypoint_data = keypoint_data
        self.imu_data = imu_data
        self.sliding_windows = sliding_window
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.participants = participants

    def run_experiment(self):
        for cv in self.participants:
            print(f'Cross Validation: {cv}')
            dataset_train = DataSet_VTT(self.keypoint_data, self.imu_data, self.sliding_windows, cv, flag='train')
            train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

            cv_save_path = os.path.join(self.save_path, f'cv_{cv}')
            if not os.path.exists(cv_save_path):
                os.makedirs(cv_save_path)
            cv_model_path = os.path.join(cv_save_path, 'saved_model.pth')
            # check if model already exists at save path
            if os.path.exists(cv_model_path):
                print('Model already exists at save path. Loading...')
                # load the model
                model_path = cv_model_path
                model = EncoderDecoder(self.input_size, self.output_size)
                model.load_state_dict(torch.load(model_path))
            else:
                model = EncoderDecoder(self.input_size, self.output_size)

            model = model.double().cuda()
            if self.loss_function == 'cross_entropy':
                criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
            else:
                criterion = nn.MSELoss(reduction="mean").to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            for epoch in range(self.num_epochs):
                print(f'Epoch: {epoch}')
                for i, data in enumerate(train_loader):
                    keypoint, imu = data
                    keypoint = keypoint.double().to(self.device)
                    imu = imu.double().to(self.device)
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
            torch.save(model.state_dict(), cv_model_path)

    def run_test(self):
        loss_per_cv = {}
        for cv in self.participants:
            print(f'Cross Validation: {cv}')
            dataset_test = DataSet_VTT(self.keypoint_data, self.imu_data, self.sliding_windows, cv, flag='test')
            test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0)

            cv_save_path = os.path.join(self.save_path, f'cv_{cv}')
            if not os.path.exists(cv_save_path):
                os.makedirs(cv_save_path)
            cv_model_path = os.path.join(cv_save_path, 'saved_model.pth')
            # check if model already exists at save path
            if os.path.exists(cv_model_path):
                print('Model exists at save path....')
                # load the model
                model_path = cv_model_path
                model = EncoderDecoder(self.input_size, self.output_size)
                model.load_state_dict(torch.load(model_path))
            else:
                print('Model does not exists at save path. Exiting....')
                return

            model = model.double().cuda()
            if self.loss_function == 'cross_entropy':
                criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
            else:
                criterion = nn.MSELoss(reduction="mean").to(self.device)

            all_loss = []
            imu = None
            output = None
            for i, data in enumerate(test_loader):
                keypoint, imu = data
                keypoint = keypoint.double().to(self.device)
                imu = imu.double().to(self.device)
                output = model(keypoint)
                output = output.squeeze(1)
                loss = criterion(output, imu)
                all_loss.append(loss.item())

            # plot the imu and predicted imu
            plt.plot(imu.cpu().detach().numpy()[1], label='IMU')
            plt.plot(output.cpu().detach().numpy()[1], label='Predicted IMU')
            plt.legend()
            #save the plot
            plt.savefig(f'../graphs/imu_plot{cv}.png')
            plt.show()

            mean_loss = sum(all_loss) / len(all_loss)
            print(f'Mean Loss: {mean_loss}')
            loss_per_cv[cv] = mean_loss
        return loss_per_cv

