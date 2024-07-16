import torch
import torch.nn as nn
import numpy as np
import os


# 1. 数据预处理
class RadarDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_paths = self.get_all_data_paths()

    def get_all_data_paths(self):
        paths = []
        for variable in ["dBZ", "ZDR", "KDP"]:
            for height in ["1.0km", "3.0km", "7.0km"]:
                dir_path = os.path.join(self.root_dir, variable, height)
                for folder in os.listdir(dir_path):
                    folder_path = os.path.join(dir_path, folder)
                    if os.path.isdir(folder_path):
                        paths.append(folder_path)
        return paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        frames = sorted(os.listdir(data_path))

        # Loading the data
        input_data = []
        for frame in frames[:10]:
            frame_path = os.path.join(data_path, frame)
            data = np.load(frame_path)
            input_data.append(data)

        target_path = os.path.join(self.root_dir, "dBZ", "1.0km", os.path.basename(data_path), frames[10])
        target_data = [np.load(target_path) for frame in frames[10:20]]

        input_data = np.stack(input_data, axis=0)  # Shape: (10, 3, 256, 256)
        target_data = np.stack(target_data, axis=0)  # Shape: (10, 1, 256, 256)

        return torch.FloatTensor(input_data), torch.FloatTensor(target_data)


# 2. 构建ConvLSTM模型
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x):
        layer_output_list = []
        last_state_list = []

        batch_size, sequence_length, _, height, width = x.size()
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = torch.zeros(batch_size, self.hidden_dim, height, width).cuda(), \
                   torch.zeros(batch_size, self.hidden_dim, height, width).cuda()

            output_inner = []
            for t in range(sequence_length):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list[-1], last_state_list[-1]


# 3. 定义损失函数和优化器
model = ConvLSTM().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
dataset = RadarDataset(root_dir="root_dir")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

num_epochs = 50
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
