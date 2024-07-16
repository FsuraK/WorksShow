import torch, os
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


seq_len = 1
norm_param = {
            'dBZ': [0, 65],
            'ZDR': [-1, 5],
            'KDP': [-1, 6],
            'rain': [0, 100]}


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
            for height in ["1.0km"]:
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


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Pointwise convolution to produce query, key, and value
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Compute query, key, value
        Q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C'
        K = self.key(x).view(batch_size, -1, width * height)  # B x C' x N
        V = self.value(x).view(batch_size, -1, width * height)  # B x C x N

        # Compute attention weights
        attn_weights = torch.bmm(Q, K)  # B x N x N
        attn_weights = torch.softmax(attn_weights / (C ** 0.5), dim=-1)

        # Compute attended values
        attn_values = torch.bmm(V, attn_weights.permute(0, 2, 1))  # B x C x N
        attn_values = attn_values.view(batch_size, C, width, height)

        # Return weighted sum of original feature map and attention map
        return x + self.gamma * attn_values


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

        # Assign network parts to specific GPUs

        # Encoder part 1 - GPU 1
        self.enc1 = nn.Sequential(
            ResidualBlock(2, 32),
            # SelfAttention(32),
            nn.MaxPool2d(2)
        ).to('cuda:1')

        # Encoder part 2 - GPU 2
        self.enc2 = nn.Sequential(
            ResidualBlock(32, 64),
            # SelfAttention(64),
            nn.MaxPool2d(2)
        ).to('cuda:2')

        # Middle part - GPU 3
        self.middle = nn.Sequential(
            ResidualBlock(64, 128),
            SelfAttention(128),
            ResidualBlock(128, 128)
        ).to('cuda:3')

        # Decoder - GPU 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ResidualBlock(64, 64),
            # SelfAttention(64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ResidualBlock(32, 16),
            nn.Conv2d(16, 1, kernel_size=1)
        ).to('cuda:4')

    def forward(self, x):
        # Encoding part 1 - GPU 1
        x = x.to('cuda:1')
        x = self.enc1(x)

        # Transfer to GPU 2
        x = x.to('cuda:2')
        x = self.enc2(x)

        # Transfer to GPU 3 for the middle part
        x = x.to('cuda:3')
        x = self.middle(x)

        # Transfer to GPU 4 for decoding
        x = x.to('cuda:4')
        x = self.decoder(x)

        return x


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
batch_size = 6
epochs = 500
channels = 3

# Load dataset
dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model = RainfallEstimationCNN()
criterion = nn.MSELoss()
ssim_loss = SSIMLoss(window_size=11, size_average=True, channel=1)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
tq_bar = tqdm(range(epochs))
# Training
mini_loss, loss_mean, init_loss = np.inf, np.inf, np.inf
model_save_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q3/model_path/model_q3.pth'
model_load_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q3/model_path/model_q3_base_v3_1km_done1.pth'
model_continue_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q3/model_path/' \
                      r'model_q3_base_v3_1km_done1.pth'
# train = True
train = False
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
            outputs = model(data)
            # loss = ssim_loss(outputs, target)
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
    model.eval()

    input_data, target_data = dataset[20]
    input_data = input_data.unsqueeze(0).to(device)
    target_data = target_data.unsqueeze(0).to(device)

    outputs = model(input_data)
    target_output = outputs
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
    df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_output2.xlsx"
                , index=False, header=False)
    df = pd.DataFrame(target_data[-1].cpu().detach().numpy())
    df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_true2.xlsx"
                , index=False, header=False)
