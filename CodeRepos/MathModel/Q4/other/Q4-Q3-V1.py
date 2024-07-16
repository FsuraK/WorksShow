import torch, os
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm


seq_len = 1
norm_param = {
            'dBZ': [0, 65],
            'ZDR': [-1, 5],
            'KDP': [-1, 6],}


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
        for variable in ["dBZ", "ZDR"]:
            # for height in ["1.0km", "3.0km", "7.0km"]:
            for height in ["3.0km"]:
                data_path = os.path.join(self.root_dir, variable, height, dir_name)
                frames = sorted(os.listdir(data_path))
                if len(frames) < 2:
                    return 0, 0
                # here now only for first 0-20, can set to [a, a+20], randomly
                for frame in frames[:1]:
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
        if len(frames) < 2:
            return 0, 0
        # here now only for first 0-20, can set to [a, a+20], randomly
        for frame in frames[:1]:
            frame_path = os.path.join(data_path, frame)
            data = np.load(frame_path)
            mmin, mmax = norm_param['dBZ']
            data = (data - mmin) / (mmax - mmin)
            target_data.append(data)

        input_data = np.stack(input_data, axis=0).reshape(2, 256, 256)  # Shape: (2, 256, 256)
        target_data = np.stack(target_data, axis=0)  # Shape: (1, 256, 256)

        return torch.FloatTensor(input_data), torch.FloatTensor(target_data)


# class RainfallEstimationCNN(nn.Module):
#     def __init__(self):
#         super(RainfallEstimationCNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # 2 channels for ZH and ZDR
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(128, 1, kernel_size=3, padding=1)  # 1 channel for rainfall
#
#         self.relu = nn.ReLU()
#         self.bn32 = nn.BatchNorm2d(32)
#         self.bn64 = nn.BatchNorm2d(64)
#         self.bn128 = nn.BatchNorm2d(128)
#
#     def forward(self, x):
#         x = self.relu(self.bn32(self.conv1(x)))
#         x = self.relu(self.bn64(self.conv2(x)))
#         x = self.relu(self.bn128(self.conv3(x)))
#         x = self.conv4(x)
#         return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Handle case where input and output channels are different
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              padding=0) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.skip:
            residual = self.skip(x)

        out += residual
        out = self.relu(out)

        return out


class RainfallEstimationCNN(nn.Module):
    def __init__(self):
        super(RainfallEstimationCNN, self).__init__()

        self.res_block1 = ResidualBlock(2, 32)  # 2 channels for ZH and ZDR
        self.res_block2 = ResidualBlock(32, 64)
        self.res_block3 = ResidualBlock(64, 128)
        self.res_block4 = ResidualBlock(128, 256)
        self.res_block5 = ResidualBlock(256, 256)
        self.conv_output = nn.Conv2d(256, 1, kernel_size=3, padding=1)  # 1 channel for rainfall

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.conv_output(x)
        return x


# Hyperparameters
learning_rate = 1e-5
batch_size = 4
epochs = 200
channels = 3

# Load dataset
dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

model = RainfallEstimationCNN().to(device)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
tq_bar = tqdm(range(epochs))
# Training
mini_loss, loss_mean, init_loss = np.inf, np.inf, np.inf
model_save_path = r'/Q3/model_path/model_q3.pth'
model_load_path = r'/Q3/model_path/model_q3_base.pth'
train = True
# train = False
if train:
    for epoch in tq_bar:
        loss_record = []
        for batch_idx, (data, target) in enumerate(dataloader):
            tq_bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            if data is False:
                print("data valid! continue")
                continue
            data, target = data.to(device), target.to(device)

            # forward
            outputs = model(data)
            loss = criterion(outputs[0][0], target)

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
            now_step_with_steps = batch_idx+1/len(dataloader)
            tq_bar.set_postfix({'step': f'{now_step_with_steps:.0f}',
                                'init_loss': f'{init_loss:.5f}',
                                'BEST': f'{mini_loss:.5f}',
                                'last_step_loss': f'{loss_mean:.5f}'})
            # if (batch_idx+1) % 1 == 0:
            #     print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    print("Finished Training")
else:
    model.load_state_dict(torch.load(model_load_path))
    # model.eval()

    input_data, target_data = dataset[3]
    input_data = input_data.unsqueeze(0).to(device)
    target_data = target_data.unsqueeze(0).to(device)

    _, outputs = model(input_data)
    target_output = outputs[0][0]
    loss = criterion(target_output, target_data)
    print(loss)
    # -------------------------------
    target_data = target_data[0]
    target_output = target_output[0]

    mmin, mmax = norm_param['dBZ']
    target_output = target_output * (mmax - mmin) + mmin
    target_data = target_data * (mmax - mmin) + mmin

    # 1 vs 1
    # df = pd.DataFrame(target_output[-1].cpu().detach().numpy())
    # df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/target_output.xlsx"
    #             , index=False, header=False)
    # df = pd.DataFrame(target_data[-1].cpu().detach().numpy())
    # df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/target_data.xlsx"
    #             , index=False, header=False)
