import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from util.ConvLstmAttention import ConvLSTM
from tqdm import tqdm
import pandas as pd


norm_param = {
            'dBZ': [0, 65],
            'ZDR': [-1, 5],
            'KDP': [-1, 6],
            'rain': [0, 70]}
variables = ["dBZ", "ZDR", "KDP"]
variables_2 = ["dBZ", "ZDR"]
heights = ["1.0km", "3.0km", "7.0km"]
heights_2 = ["1.0km", "3.0km"]
height_3 = ["3.0km"]
seq_len = 3

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
        random_able = False
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
                if random_able:
                    number_ = np.random.randint(0, frame_len - 2 * seq_len + 1)
                else:
                    number_ = 0

                if variable is variables[0] and height is height_3[0]:
                    number = number_

                for frame in frames[number: seq_len + number]:
                    frame_path = os.path.join(data_path, frame)
                    data = np.load(frame_path)
                    mmin, mmax = norm_param[variable]
                    data = (data - mmin) / (mmax - mmin)
                    input_data.append(data)

        target_data = []
        rain_dir = "/home/liuyangyang/mathModelDataSet/NJU_CPOL_kdpRain"
        data_path = os.path.join(rain_dir, dir_name)
        frames = sorted(os.listdir(data_path))
        if len(frames) < 2:
            return 0, 0
        for frame in frames[number + seq_len - 1: number + seq_len]:
            frame_path = os.path.join(data_path, frame)
            data = np.load(frame_path)
            mmin, mmax = norm_param['rain']
            data = (data - mmin) / (mmax - mmin)
            target_data.append(data)

        input_data = np.stack(input_data, axis=0).reshape(seq_len, 3, 256, 256)
        target_data = np.stack(target_data, axis=0)  # Shape: (10, 1, 256, 256)

        return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

# Hyperparameters
learning_rate = 5e-7
batch_size = 4
epochs = 100
channels = 3

# Load dataset
dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model = ConvLSTM(input_dim=9, hidden_dim=64, kernel_size=(3,3), num_layers=3).to(device)

model = ConvLSTM(input_dim=channels,
                 hidden_dim=[64, 64, 1],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False).to(device)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
tq_bar = tqdm(range(epochs))
# Training
mini_loss, loss_mean, init_loss = np.inf, np.inf, np.inf
model_save_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q4/model_path/model_q2.pth'
model_load_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q4/model_path/model_q4_step6.pth'
# train = True
train = False
if train:
    # model.load_state_dict(torch.load(model_load_path))
    for epoch in tq_bar:
        loss_record = []
        for batch_idx, (data, target) in enumerate(dataloader):
            tq_bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            if data is False:
                print("data valid! continue")
                continue
            data, target = data.to(device), target.to(device)

            # forward
            _, outputs = model(data)
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
                if epoch >= 0:
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
    model.load_state_dict(torch.load(model_save_path))
    # model.eval()

    input_data, target_data = dataset[0]
    input_data = input_data.unsqueeze(0).to(device)
    target_data = target_data.unsqueeze(0).to(device)

    _, outputs = model(input_data)
    target_output = outputs[0][0]
    loss = criterion(target_output, target_data)
    print(loss)
    # -------------------------------
    target_data = target_data[0]
    target_output = target_output[0]
    input_data = input_data[0]
    mmin, mmax = norm_param['rain']
    target_output = target_output * (mmax - mmin) + mmin
    target_data = target_data * (mmax - mmin) + mmin

    # for i in range(3):  # frame
    #     input_da = input_data[i]
    #     for var in ["dBZ", "ZDR", "KDP"]:  # channel
    #         if var=="dBZ":
    #             j=0
    #         elif var=="ZDR":
    #             j=1
    #         else:
    #             j=2
    #         input_d = input_da[j]
    #         mmin, mmax = norm_param[var]
    #         input_d = input_d * (mmax - mmin) + mmin
    #         input_d = input_d.cpu().detach().numpy()
    #
    #         df = pd.DataFrame(input_d)
    #         df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q1'
    #                     + f"/input_{var[0:3]}_time{i+1}.xlsx")

    for i in range(1):
        target_out = target_output[i].cpu().detach().numpy()
        target_da = target_data[i].cpu().detach().numpy()

        df = pd.DataFrame(target_out)
        df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q4' + f"/future_pre_{i + 2}.xlsx"
                    , index=False, header=False)
        df = pd.DataFrame(target_da)
        df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q4' + f"/future_true_{i + 2}.xlsx"
                    , index=False, header=False)

