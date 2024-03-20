# import the necessary packages
import os

import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataloaders.VTT_dataloader import DataSet_VTT
from models.EncoderDecoderModel import EncoderDecoder

pd.options.display.float_format = '{:.2f}'.format

# check if cuda is available
print(torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

data_path = '../data/VTT_ConIot_Dataset'
IMU_path = data_path + '/IMU'
Keypoint_path = data_path + '/Keypoint'
activities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
users = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def create_keypoint_and_imu_data(subject, activity):
    # check if file exists
    if not os.path.exists(Keypoint_path + f'/Subject_{subject:02d}_Task_{activity}.m2ts_keyPoints.csv'):
        print(f'Keypoint file for Subject_{subject:02d}_Task_{activity} does not exist')
        # return two empty dataframes
        return pd.DataFrame(), pd.DataFrame()

    keypoint_data = pd.read_csv(Keypoint_path + f'/Subject_{subject:02d}_Task_{activity}.m2ts_keyPoints.csv')
    imu_data = pd.read_csv(IMU_path + f'/activity_{activity}_user_{subject}_combined.csv')
    # only keep the columns in imu data that are of accelerometer, i.e. that have _A in the name
    imu_data = imu_data[[col for col in imu_data.columns if '_A' in col]]
    # remove frame_number and timestamp columns from keypoint data
    keypoint_data = keypoint_data.drop(columns=['frame_number', 'timestamp'])
    keypoint_data['subject'] = subject
    keypoint_data['activity'] = activity
    imu_data['subject'] = subject
    imu_data['activity'] = activity
    # make sure subject and activity are of type float
    keypoint_data['subject'] = keypoint_data['subject'].astype(float)
    keypoint_data['activity'] = keypoint_data['activity'].astype(float)
    imu_data['subject'] = imu_data['subject'].astype(float)
    imu_data['activity'] = imu_data['activity'].astype(float)
    return keypoint_data, imu_data

def create_relative_keypoints(df_keypoints):
    '''
    "keypoints": [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"
        ],
    "corresponding points":[
            "0", "1", "2", "3", "4",
            "5", "6", "7", "8",
            "9", "10", "11", "12",
            "13", "14", "15", "16"
        ]
    There are 17 points with each point having their x, y and confidence values
    :param df_keypoints:
    :return:
    '''
    # drop the column "detection_score" as they are all high
    df_keypoints = df_keypoints.drop(columns=['detection_score'])
    # create a weighted average column of the x and y of nose, left eye, right eye, left ear and right ear and call it x_head, y_head, prob_head
    # where the columns for x and y are labelled as x0,y0,prob0,x1,y1,prob1,....,x16,y16,prob16
    # get the x and y values for the nose, left eye, right eye, left ear and right ear
    x_head = (df_keypoints['x0'] * df_keypoints['prob0'] + df_keypoints['x1'] * df_keypoints['prob1'] +
              df_keypoints['x2'] * df_keypoints['prob2'] + df_keypoints['x3'] * df_keypoints['prob3'] +
              df_keypoints['x4'] * df_keypoints['prob4']) / (df_keypoints['prob0'] + df_keypoints['prob1'] +
                                                             df_keypoints['prob2'] + df_keypoints['prob3'] +
                                                             df_keypoints['prob4'])

    y_head = (df_keypoints['y0'] * df_keypoints['prob0'] + df_keypoints['y1'] * df_keypoints['prob1'] +
              df_keypoints['y2'] * df_keypoints['prob2'] + df_keypoints['y3'] * df_keypoints['prob3'] +
              df_keypoints['y4'] * df_keypoints['prob4']) / (df_keypoints['prob0'] + df_keypoints['prob1'] +
                                                             df_keypoints['prob2'] + df_keypoints['prob3'] +
                                                             df_keypoints['prob4'])
    # set prob_head as the maximum of the probabilities of the nose, left eye, right eye, left ear and right ear
    prob_head = df_keypoints[['prob0', 'prob1', 'prob2', 'prob3', 'prob4']].max(axis=1)

    # add the columns to the dataframe
    df_keypoints['x_head'] = x_head
    df_keypoints['y_head'] = y_head
    df_keypoints['prob_head'] = prob_head

    # drop the columns for the nose, left eye, right eye, left ear and right ear
    df_keypoints = df_keypoints.drop(columns=['x0', 'y0', 'prob0', 'x1', 'y1', 'prob1', 'x2', 'y2', 'prob2',
                                              'x3', 'y3', 'prob3', 'x4', 'y4', 'prob4'])

    # convert the x and y values to be relative to the head
    for i in range(5, 17):
        df_keypoints[f'x{i}'] = df_keypoints[f'x{i}'] - df_keypoints['x_head']
        df_keypoints[f'y{i}'] = df_keypoints[f'y{i}'] - df_keypoints['y_head']

    # drop the columns for the x and y values of the head
    df_keypoints = df_keypoints.drop(columns=['x_head', 'y_head', 'prob_head'])

    return df_keypoints


def processed_data():
    full_keypoint_data = pd.DataFrame()
    full_imu_data = pd.DataFrame()
    for activity in activities:
        for user in users:
            keypoint_data, imu_data = create_keypoint_and_imu_data(user, activity)
            full_keypoint_data = pd.concat([full_keypoint_data, keypoint_data])
            full_imu_data = pd.concat([full_imu_data, imu_data])

    # pre-pocess by removing the extra time stamps fom either keypoint or imu data
    # per participant and activity pair there should be 4x the number of imu data points as keypoint data points
    cropped_keypoint_data = pd.DataFrame()
    cropped_imu_data = pd.DataFrame()
    for activity in activities:
        for user in users:
            # print shape of keypoint and imu data
            keypoint_data = full_keypoint_data[
                (full_keypoint_data['subject'] == user) & (full_keypoint_data['activity'] == activity)]
            imu_data = full_imu_data[(full_imu_data['subject'] == user) & (full_imu_data['activity'] == activity)]

            # if either of the data is empty, skip
            if keypoint_data.shape[0] == 0 or imu_data.shape[0] == 0:
                continue

            if keypoint_data.shape[0] * 4 > imu_data.shape[0]:
                # remove the extra keypoint data
                keypoint_data = keypoint_data.iloc[:imu_data.shape[0] // 4]
            else:
                # remove the extra imu data
                imu_data = imu_data.iloc[:keypoint_data.shape[0] * 4]
            # remove keypoint values such that keypoint data is multiple of 25
            keypoint_data = keypoint_data.iloc[:keypoint_data.shape[0] // 25 * 25]

            # remove imu values such that imu data is multiple of 100
            imu_data = imu_data.iloc[:imu_data.shape[0] // 100 * 100]

            # make copy of the columns subject and activity before dropping
            subject_col = imu_data['subject']
            activity_col = imu_data['activity']
            imu_data = imu_data.drop(columns=['subject', 'activity'])
            # resample the imu data to be the same length as the keypoint data
            imu_resampled = resample(imu_data, keypoint_data.shape[0])
            # create a pd dataframe from the resampled data using the columns in imu_data
            imu_data = pd.DataFrame(imu_resampled, columns=imu_data.columns)
            # add the subject and activity columns back
            imu_data['subject'] = subject_col
            imu_data['activity'] = activity_col

            cropped_keypoint_data = pd.concat([cropped_keypoint_data, keypoint_data])
            cropped_imu_data = pd.concat([cropped_imu_data, imu_data])

    # reset the index keypoint
    cropped_keypoint_data = cropped_keypoint_data.reset_index(drop=True)
    # preprocess the keypoints for each point of interest from pose estimation
    cropped_keypoint_data = create_relative_keypoints(cropped_keypoint_data)

    # reset the index for imu
    cropped_imu_data = cropped_imu_data.reset_index(drop=True)

    # create a new df between 25 key-points and 100 IMU data points for each activity and user
    # with columns start, end, subject, activity
    sliding_windows = pd.DataFrame(columns=['start', 'end', 'subject', 'activity'])

    for activity in activities:
        for user in users:
            # preserve the index
            keypoint_data = cropped_keypoint_data[
                (cropped_keypoint_data['subject'] == user) & (cropped_keypoint_data['activity'] == activity)]
            # if either of the data is empty, skip
            if keypoint_data.shape[0] == 0:
                continue

            # split the data into windows of 25
            for i in range(0, keypoint_data.shape[0], 25):
                # check if the window has 25 elements
                if i + 25 > keypoint_data.shape[0]:
                    continue
                # get the start and end index of the window in cropped_keypoint_data
                start = keypoint_data.index[i]
                end = keypoint_data.index[i + 25 - 1]
                # concat to the sliding_windows dataframe
                sliding_windows = pd.concat([sliding_windows, pd.DataFrame(
                    {'start': [start], 'end': [end], 'subject': [user], 'activity': [activity]})])

    sliding_windows = sliding_windows.reset_index(drop=True)

    # normalize the keypoint data
    # create a copy of column activity and subject before dropping
    activity = cropped_keypoint_data['activity']
    subject = cropped_keypoint_data['subject']
    # create a copy of all columns starting with prob
    prob_cols = cropped_keypoint_data.filter(regex='prob').copy()
    # drop the columns activity, subject and all columns starting with prob
    cropped_keypoint_data = cropped_keypoint_data.drop(columns=['activity', 'subject'])
    cropped_keypoint_data = cropped_keypoint_data.drop(columns=cropped_keypoint_data.filter(regex='prob').columns)
    # Do a normalization
    scaler = StandardScaler()
    cropped_keypoint_data = pd.DataFrame(scaler.fit_transform(cropped_keypoint_data),
                                         columns=cropped_keypoint_data.columns)
    # add the columns activity and subject back
    cropped_keypoint_data['activity'] = activity
    cropped_keypoint_data['subject'] = subject
    # add the columns starting with prob back
    cropped_keypoint_data = pd.concat([cropped_keypoint_data, prob_cols], axis=1)

    return cropped_keypoint_data, cropped_imu_data, sliding_windows

# main method
if __name__ == '__main__':
    # get the processed data
    keypoint_data, imu_data, sliding_windows = processed_data()

    # replace any nan value as the mean of the value before and after it
    keypoint_data = keypoint_data.fillna(keypoint_data.mean())
    imu_data = imu_data.fillna(imu_data.mean())

    dataset_train = DataSet_VTT(keypoint_data, imu_data, sliding_windows, flag='train')
    dataset_test = DataSet_VTT(keypoint_data, imu_data, sliding_windows, flag='test')
    train_loader = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (25, 36)
    output_size = (25, 9)
    model = EncoderDecoder(input_size, output_size)
    model = model.double().cuda()
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(150):
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
    torch.save(model.state_dict(), 'encoder_decoder_model.pth')