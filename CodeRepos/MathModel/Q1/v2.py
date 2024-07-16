import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from util.ConvLstm import ConvLSTM
from tqdm import tqdm
import pandas as pd


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

        for variable in ["dBZ", "ZDR", "KDP"]:
            # for height in ["1.0km", "3.0km", "7.0km"]:
            for height in ["3.0km"]:
                data_path = os.path.join(self.root_dir, variable, height, dir_name)
                frames = sorted(os.listdir(data_path))
                if len(frames)<20:
                    return 0, 0
                # here now only for first 0-20, can set to [a, a+20], randomly
                for frame in frames[:3]:
                    frame_path = os.path.join(data_path, frame)
                    data = np.load(frame_path)
                    mmin, mmax = norm_param[variable]
                    data = (data - mmin) / (mmax - mmin)
                    input_data.append(data)

        target_data = []
        for frame in frames[5:10]:
            frame_path = os.path.join(self.root_dir, "dBZ", "3.0km", dir_name, frame)
            data = np.load(frame_path)
            mmin, mmax = norm_param['dBZ']
            data = (data - mmin) / (mmax - mmin)
            target_data.append(data)

        # print(len(input_data))
        # input_data = np.stack(input_data, axis=0).reshape(10, 9, 256, 256)  # Shape: (10, 9, 256, 256)
        input_data = np.stack(input_data, axis=0).reshape(10, 3, 256, 256)  # Shape: (10, 3, 256, 256)
        target_data = np.stack(target_data, axis=0)  # Shape: (10, 1, 256, 256)

        return torch.FloatTensor(input_data), torch.FloatTensor(target_data)


learning_rate = 1e-5
batch_size = 4
epochs = 50
channels = 3

# Load dataset
dataset = RadarDataset("/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model = ConvLSTM(input_dim=9, hidden_dim=64, kernel_size=(3,3), num_layers=3).to(device)

model = ConvLSTM(input_dim=channels,
                 hidden_dim=[32, 32, 10],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False).to(device)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
tq_bar = tqdm(range(epochs))
# Training
mini_loss, loss_mean, init_loss = np.inf, np.inf, np.inf
model_save_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q1/model_path/model_q1.pth'
model_load_path = r'/home/liuyangyang/MathModel/math-model-general/Code/Q1/model_path/model_q1_base.pth'
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
    mmin, mmax = norm_param['dBZ']
    target_output = target_output * (mmax - mmin) + mmin
    target_data = target_data * (mmax - mmin) + mmin
    input_data = input_data[0]

    for i in range(3):
        target_output = target_output[i]
        target_data = target_data[i]

        df = pd.DataFrame(target_output[-1].cpu().detach().numpy())
        df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q1' + f"/future_pre_{i+1}.xlsx"
                    , index=False, header=False)
        df = pd.DataFrame(target_data[-1].cpu().detach().numpy())
        df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test/q1' + f"/future_true_{i+1}.xlsx"
                    , index=False, header=False)




