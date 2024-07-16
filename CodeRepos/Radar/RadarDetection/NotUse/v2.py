import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

""" chosen different model class based on the 4 question """
from RadarDetection.algorithms.conv_lstm import ConvLSTM
from tqdm import tqdm
import pandas as pd


norm_param = {
            'dBZ': [0, 65],
            'ZDR': [-1, 5],
            'KDP': [-1, 6],
            'rain': [0, 100]}
variables = ["dBZ", "ZDR", "KDP"]
variables_2 = ["dBZ", "ZDR", "KDP"]
heights = ["1.0km", "3.0km", "7.0km"]
heights_2 = ["1.0km", "3.0km"]
height_3 = ["3.0km"]

""" chose 10 for question 1 and  2, chose 1 for question 3 """
seq_len = 2


# load data / chose data
class RadarDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_dirs = sorted([name for name in os.listdir(os.path.join(root_dir, "dBZ", "3.0km")) if
                                 os.path.isdir(os.path.join(root_dir, "dBZ", "3.0km", name))])

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        dir_name = self.data_dirs[idx]
        input_data = []
        frame_len = None
        number = 0
        """ variables for question 3, variables for questions 1, 2"""
        for variable in variables:
            # for height in ["1.0km", "3.0km", "7.0km"]:
            # for data in ["test", "validate"]
            """chose different height for training, or use them all"""
            for height in height_3:
                data_path = os.path.join(self.root_dir, variable, height, dir_name)
                frames = sorted(os.listdir(data_path))
                frame_len = len(frames)
                if frame_len < 2*seq_len:
                    return 0, 0

                # random sample data
                number_ = np.random.randint(0, frame_len - 2 * seq_len + 1)

                if variable is variables[0] and height is height_3[0]:
                    number = number_

                for frame in frames[number: seq_len + number]:
                    frame_path = os.path.join(data_path, frame)
                    data = np.load(frame_path)
                    mmin, mmax = norm_param[variable]
                    data = (data - mmin) / (mmax - mmin)
                    input_data.append(data)

        target_data = []
        for frame in frames[number + seq_len: number + 2*seq_len]:
            frame_path = os.path.join(self.root_dir, variables[0], heights[1], dir_name, frame)
            data = np.load(frame_path)
            mmin, mmax = norm_param[variables[0]]
            data = (data - mmin) / (mmax - mmin)
            target_data.append(data)

        """ different for questions 1-4 """
        # Shape: (10, 3, 256, 256) --> question 1 and 2
        input_data = np.stack(input_data, axis=0).reshape(seq_len, 3, 256, 256)
        target_data = np.stack(target_data, axis=0)  # Shape: (10, 1, 256, 256)
        # Shape: (10, 3, 256, 256) --> question 3 and 4
        # input_data = np.stack(input_data, axis=0).reshape(2, 256, 256)  # Shape: (2, 256, 256)
        # target_data = np.stack(target_data, axis=0)  # Shape: (1, 256, 256)

        # # # # questions 3, 4
        # rain_dir = "/home/liuyangyang/mathModelDataSet/NJU_CPOL_kdpRain"
        # data_path = os.path.join(rain_dir, dir_name)
        # frames = sorted(os.listdir(data_path))
        # if len(frames) < 2:
        #     return 0, 0
        # # here now only for first 0-20, can set to [a, a+20], randomly
        # for frame in frames[number + seq_len - 1: number + seq_len]:
        #     frame_path = os.path.join(data_path, frame)
        #     data = np.load(frame_path)
        #     # if train is False:
        #     #     df = pd.DataFrame(data)
        #     #     df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_raw.xlsx"
        #     #                 , index=False, header=False)
        #     # mmin, mmax = norm_param['dBZ']
        #     mmin, mmax = norm_param['dBZ']
        #     data = (data - mmin) / (mmax - mmin)
        #     target_data.append(data)

        return torch.FloatTensor(input_data), torch.FloatTensor(target_data)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size)

    def gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.Tensor([(x - window_size // 2) ** 2 for x in range(window_size)]) / (-2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)  # 将窗口移到与img1相同的设备上

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


# Hyperparameters
learning_rate = 5e-6
batch_size = 1
epochs = 500
channels = 3

""" Load dataset based on questions 1-4 and test and validate """
dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308")
# dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308_test")
# dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308_validate")
# dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_kdpRain")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

""" chose different model in 附录2 for different questions (1 - 4) """
model = ConvLSTM(input_dim=channels,
                 hidden_dim=[32, 32, seq_len],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False).to(device)
# model = LightweightSwinTransformer(channels, 1).to(device)
criterion = nn.MSELoss()
criterion_add = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer_add = SSIMLoss(window_size=11, size_average=True, channel=1)
# optimizer_add_add = optim.SGD(model.parameters(), lr=learning_rate)
tq_bar = tqdm(range(epochs))
# Training
mini_loss, loss_mean, init_loss = np.inf, np.inf, np.inf

# model path save / load / continue
model_save_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q1/model_path/model_q1.pth'
model_load_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q1/model_path/model_q1_step10.pth'
model_continue_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q1/model_path/model_q1_step10.pth'

train = True
# train = False
# continue_train = True
continue_train = False
if train:
    if continue_train:
        model.load_state_dict(torch.load(model_continue_path))
    for epoch in tq_bar:
        loss_record = []
        for batch_idx, (data, target) in enumerate(dataloader):
            tq_bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            if data is False:
                print("data valid! continue")
                continue
            data, target = data.to(device), target.to(device)

            # forward
            # input.shape = torch.Size([1, 1, 1, 256, 256]) = [batch_size, seq_len, channels, w, h] 1 2 3 256 256
            # outputs[0][0].shape = torch.Size([1, 1, 256, 256]) = [batch_size, seq_len, w, h]      1 2 256 256
            _, outputs = model(data)

            # loss function
            loss_MSE = criterion(outputs[0][0], target)
            loss_SIMM = criterion_add(outputs[0][0], target)
            loss = 0.5 * loss_SIMM + 0.5 * loss_MSE  # only loss_MSE for question 1
            # loss = loss_MSE

            # backward and optimize here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print info
            loss_record.append(loss.item())
            if len(loss_record) > 10:
                loss_mean = np.mean(loss_record[-10:])
            elif len(loss_record) == 10 and epoch == 0:
                init_loss = np.mean(loss_record[-10:])
            # Init_reward = ep_mean_reward if episode == 10 else 0
            # bf_reward = episode_reward if episode_reward <= bf_reward else bf_reward
            if loss_mean <= mini_loss:
                mini_loss = loss_mean
                if epoch >= 1:
                    torch.save(model.state_dict(), model_save_path)
            now_step_with_steps = batch_idx+1/len(dataloader)
            tq_bar.set_postfix({'step': f'{now_step_with_steps:.0f}',
                                'init_loss': f'{init_loss:.5f}',
                                'BEST': f'{mini_loss:.5f}',
                                'last_step_loss': f'{loss_mean:.5f}'})

    print("Training done")
else:
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    input_random = True
    # input_random = False

    if input_random:
        input_dir = np.random.randint(0, 256 + 1)
        input_data, target_data = dataset[input_dir]
    else:
        input_data, target_data = dataset[0]

    input_data = input_data.unsqueeze(0).to(device)
    target_data = target_data.unsqueeze(0).to(device)

    _, outputs = model(input_data)
    target_output = outputs[0][0]
    loss = criterion(target_output, target_data)
    print(loss)

    # get target data from the model output, get true data, in a lower dimension
    target_data = target_data[0]
    target_output = target_output[0]
    input_data = input_data[0]

    """ select different target based on th chosen trained model """
    mmin, mmax = norm_param['dBZ']
    target_output = target_output * (mmax - mmin) + mmin
    target_data = target_data * (mmax - mmin) + mmin

    """ chosee to save predict or estimate result to excel, based on questions 1-4"""
    for i in range(3):  # frame
        input_da = input_data[i]
        for var in ["dBZ", "ZDR", "KDP"]:  # channel
            if var=="dBZ":
                j=0
            elif var=="ZDR":
                j=1
            else:
                j=2
            input_d = input_da[j]
            mmin, mmax = norm_param[var]
            input_d = input_d * (mmax - mmin) + mmin
            input_d = input_d.cpu().detach().numpy()

            df = pd.DataFrame(input_d)
            df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q1'
                        + f"/input_{var[0:3]}_time{i+1}.xlsx"
                        , index=False, header=False)

    for i in range(3):
        target_out = target_output[i].cpu().detach().numpy()
        target_da = target_data[i].cpu().detach().numpy()

        df = pd.DataFrame(target_out)
        df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q1' + f"/future_pre_{i + 1}.xlsx"
                    , index=False, header=False)
        df = pd.DataFrame(target_da)
        df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q1' + f"/future_true_{i + 1}.xlsx"
                    , index=False, header=False)

