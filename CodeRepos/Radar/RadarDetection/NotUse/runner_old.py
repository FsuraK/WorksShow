import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from RadarDetection.data.data_loader import DataLoaderSelf


norm_param = {
            'dBZ': [0, 65],
            'ZDR': [-1, 5],
            'KDP': [-1, 6],}
# Hyperparameters
learning_rate = 1e-5
epochs = 400
channels = 3

dataset, dataloader = DataLoaderSelf(config)

# Initialize model, criterion, and optimizer
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

model = RainfallEstimationCNN().to(device)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
tq_bar = tqdm(range(epochs))
# Training
mini_loss, loss_mean, init_loss = np.inf, np.inf, np.inf
model_save_path = r'/Q3/model_path/model_q3_base.pth'
model_load_path = r'/Q3/model_path/model_q3_base_v2_done.pth'
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

    input_data, target_data = dataset[1]
    input_data = input_data.unsqueeze(0).to(device)
    target_data = target_data.unsqueeze(0).to(device)

    outputs = model(input_data)
    target_output = outputs
    loss = criterion(target_output, target_data)
    print(loss)
    # -------------------------------
    target_data = target_data[0]
    target_output = target_output[0]

    mmin, mmax = norm_param['dBZ']
    target_output = target_output * (mmax - mmin) + mmin
    target_data = target_data * (mmax - mmin) + mmin

    # 1 vs 1
    df = pd.DataFrame(target_output[-1].cpu().detach().numpy())
    df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_output.xlsx"
                , index=False, header=False)
    df = pd.DataFrame(target_data[-1].cpu().detach().numpy())
    df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/test' + "/rain_true.xlsx"
                , index=False, header=False)
