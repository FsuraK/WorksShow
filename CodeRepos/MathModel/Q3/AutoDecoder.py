import random

import torch, os
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from util.BaseUtilFunc import Autoencoder

seq_len = 1
norm_param = {
    'dBZ': [0, 65],
    'ZDR': [-1, 5],
    'KDP': [-1, 6],
    'rain': [0, 85]}


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
        num = 0
        for variable in ["dBZ", "ZDR"]:
            # for height in ["1.0km", "3.0km", "7.0km"]:
            for height in ["3.0km"]:
                data_path = os.path.join(self.root_dir, variable, height, dir_name)
                frames = sorted(os.listdir(data_path))
                k = len(frames)
                if k < 2:
                    return 0, 0
                # here now only for first 0-20, can set to [a, a+20], randomly
                if variable == "dBZ":
                    num = random.randint(0, k - 1)
                for frame in frames[num: num + 1]:
                    frame_path = os.path.join(data_path, frame)
                    data = np.load(frame_path)
                    mmin, mmax = norm_param[variable]
                    data = (data - mmin) / (mmax - mmin)
                    input_data.append(data)

        # target data
        target_data = []
        rain_dir = "/home/liuyangyang/mathModelDataSet/NJU_CPOL_kdpRain"
        data_path = os.path.join(rain_dir, dir_name)
        frames = sorted(os.listdir(data_path))
        for frame in frames[num: num + 1]:
            frame_path = os.path.join(data_path, frame)
            data = np.load(frame_path)
            # if train is False:
            #     df = pd.DataFrame(data)
            #     df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_raw.xlsx"
            #                 , index=False, header=False)
            mmin, mmax = norm_param['rain']
            data = (data - mmin) / (mmax - mmin)
            target_data.append(data)

        input_data = np.stack(input_data, axis=0).reshape(2, 256, 256)  # Shape: (2, 256, 256)
        target_data = np.stack(target_data, axis=0)  # Shape: (1, 256, 256)

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

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


def add_gaussian_noise(tensor, mean=0, std=0.05):
    # Convert tensor to numpy
    data = tensor.cpu().numpy()

    # Add noise
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise

    # Convert back to tensor
    noisy_tensor = torch.from_numpy(noisy_data).float()

    return noisy_tensor
# Hyperparameters
learning_rate = 5e-6
batch_size = 6
epochs = 500
channels = 3

# Load dataset
dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cpu")

model = Autoencoder().to(device)
criterion = nn.MSELoss()
ssim_loss = SSIMLoss(window_size=11, size_average=True, channel=1)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
tq_bar = tqdm(range(epochs))
# Training
mini_loss, loss_mean, init_loss = np.inf, np.inf, np.inf
model_save_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q3/model_path/model_autoDecoder.pth'
model_load_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q3/model_path/model_autoDecoder.pth'
model_continue_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q3/model_path/model_autoDecoder.pth'

# train = True
train = False
continue_train = False
if train:
    if continue_train:
        model.load_state_dict(torch.load(model_continue_path))
    for epoch in tq_bar:
        loss_record = []
        for batch_idx, (data, target) in enumerate(dataloader):
            tq_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            if data is False:
                print("data valid! continue")
                continue
            _, target = data.to(device), target.to(device)

            # forward
            outputs = model(target)
            # loss = ssim_loss(outputs, target)
            # loss = criterion(outputs, target) + ssim_loss(outputs, target) * 0.8
            loss = criterion(outputs, target)

            # backward and optimize
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
                if epoch >= 10:
                    torch.save(model.state_dict(), model_save_path)
            now_step_with_steps = batch_idx + 1 / len(dataloader)
            tq_bar.set_postfix({'step': f'{now_step_with_steps:.0f}',
                                'init_loss': f'{init_loss:.5f}',
                                'BEST': f'{mini_loss:.5f}',
                                'last_step_loss': f'{loss_mean:.5f}'})
            # if (batch_idx+1) % 1 == 0:
            #     print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    print("Finished Training")
else:
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    _, target_data = dataset[30]
    target_data = target_data.unsqueeze(0).to(device)

    target_output = add_gaussian_noise(target_data)
    loss = criterion(target_output, target_data)
    print(loss)
    # -------------------------------
    target_data = target_data[0]
    target_output = target_output[0]

    mmin, mmax = norm_param['rain']
    target_output = target_output * (mmax - mmin) + mmin
    target_data = target_data * (mmax - mmin) + mmin

    # 1 vs 1
    df = pd.DataFrame(target_output[-1].cpu().detach().numpy())
    df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_output3.xlsx"
                , index=False, header=False)
    df = pd.DataFrame(target_data[-1].cpu().detach().numpy())
    df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_true3.xlsx"
                , index=False, header=False)
