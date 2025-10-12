import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ------------------- Dataset -------------------
class FolderTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset load từng file .npy từ folder:
    dataset_ts/train/val/test
        features/0000.npy
        target/0000.npy
    """
    def __init__(self, root_dir, time_in=30, time_out=1):
        self.feature_dir = os.path.join(root_dir, "features")
        self.target_dir  = self.feature_dir

        self.feature_files = sorted(os.listdir(self.feature_dir))
        self.target_files  = sorted(os.listdir(self.target_dir))
        assert len(self.feature_files) == len(self.target_files), "features != target"

        self.time_in = time_in
        self.time_out = time_out

        # Tạo danh sách index cho sliding window
        self.sample_indices = []
        total_timesteps = len(self.feature_files)
        for i in range(total_timesteps - time_in - time_out + 1):
            self.sample_indices.append(i)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]

        # Load input sequence
        X_list = []
        for i in range(start_idx, start_idx + self.time_in):
            feat = np.load(os.path.join(self.feature_dir, self.feature_files[i]))  # [H,W,C]
            feat = np.transpose(feat, (2,0,1))  # [C,H,W]
            X_list.append(feat)
        X = np.stack(X_list, axis=0)  # [time_in, C, H, W]

        # Load output sequence
        Y_list = []
        for i in range(start_idx + self.time_in, start_idx + self.time_in + self.time_out):
            targ = np.load(os.path.join(self.target_dir, self.target_files[i]))  # [H,W,C] (C=1)
            targ = np.transpose(targ, (2,0,1))  # [C,H,W]
            Y_list.append(targ)
        Y = np.stack(Y_list, axis=0)  # [time_out, C, H, W]

        return torch.FloatTensor(X), torch.FloatTensor(Y)

# ------------------- ConvLSTM Cell -------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        H, W = image_size
        return (torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
                torch.zeros(batch_size, self.hidden_dim, H, W, device=device))

# ------------------- ConvLSTM -------------------
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]*num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_in = input_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(ConvLSTMCell(cur_in, self.hidden_dim[i], kernel_size, bias=bias))

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1,0,2,3,4)
        B, T, C, H, W = x.size()
        if hidden_state is None:
            device = x.device
            hidden_state = [cell.init_hidden(B, (H,W), device) for cell in self.cell_list]

        layer_output_list = []
        last_state_list = []

        cur_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = self.cell_list[layer_idx](cur_input[:,t], [h,c])
                output_inner.append(h)
            layer_out = torch.stack(output_inner, dim=1)
            cur_input = layer_out
            layer_output_list.append(layer_out)
            last_state_list.append([h,c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

# ------------------- Forecaster -------------------
class ConvLSTMForecaster(nn.Module):
    def __init__(self, input_channels, hidden_dims=[64,32,16], kernel_size=3, dropout_rate=0.2):
        super().__init__()
        self.convlstm = ConvLSTM(input_channels, hidden_dims, kernel_size, len(hidden_dims),
                                 batch_first=True)
        self.batch_norm = nn.BatchNorm2d(hidden_dims[-1])
        self.dropout = nn.Dropout2d(dropout_rate)
        self.output_conv = nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        layer_out, last_state = self.convlstm(x)
        h_last = last_state[0][0]  # last hidden of last layer
        h_last = self.batch_norm(h_last)
        h_last = self.dropout(h_last)
        out = self.output_conv(h_last)
        out = out.unsqueeze(1)  # [B,1,C,H,W]
        return out