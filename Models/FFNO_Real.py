from flax import linen as nn
from typing import Sequence, Tuple
import numpy as np
from torch.utils import data
from functools import partial
import time
import scipy.io
import os
import argparse
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import pickle
from sklearn import metrics
from termcolor import colored
import random
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import rc
import seaborn as sns
import h5py
from scipy.ndimage import rotate, zoom
import nibabel as nib
from nilearn import plotting
from collections import defaultdict
import torch
#import torchvision.nn.functional as F
import torch.nn.functional as F
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 2
n_sensors = 1
branch_layers = [64, 64, 64] 
branch_input_features = 1
trunk_layers = [64, 64, 64]

t1_shape = (80, 80, 44) 
displacement_shape = (80, 80, 44, 3)

trunk_input_features = 3
hidden_dim = 100
p_test = 100
result_dir = './'
epochs = 500
vis_iter = 1000
lr = 1e-3
transition_steps = 2000
decay_rate = 0.9

t1_small_img = torch.ones((1, 80, 80, 44))  
scalar_input = torch.ones((1, 1))          
freq_in = torch.ones((1, 1))              

DIRECTION_ENCODING = {
    'AP': 1.0,
    'LR': 0.0
}

SEX_ENCODING = {
    'M': 1.0,
    'F': 0.0,
    'unknown': -1.0  
}

def load_and_preprocess_data(file_path):
    data_samples = []
    with h5py.File(file_path, 'r') as f:
        n_samples = len([k for k in f.keys() if k.startswith('sample_')])

        total_age = 0
        total_brain_volume = 0

        for i in range(n_samples):
            sample = f[f'sample_{i}']
            
            t1 = sample['t1'][:]
            t1_affine = sample['t1_affine'][:]
            mask = sample['brainmask'][:].astype(bool)
            mask_affine = sample['brainmask_affine'][:]
            freq = sample.attrs['frequency']

            direction = sample.attrs['direction']
            direction_value = DIRECTION_ENCODING[direction]

            sex = sample.attrs.get('sex', 'unknown')
            age = sample.attrs.get('age', -1)
            brain_volume = sample.attrs.get('brain_volume', -1)


            sex_value = SEX_ENCODING[sex]

            disp_real = sample['displacement/real/data'][:]
            disp_imag = sample['displacement/imag/data'][:]
            disp_re_affine = sample['displacement/real/affine'][:]
            disp_im_affine = sample['displacement/imag/affine'][:]

            x_coords = sample['disp_coords/x'][:]
            y_coords = sample['disp_coords/y'][:]
            z_coords = sample['disp_coords/z'][:]

            coords_shape = disp_real.shape[:-1] if len(disp_real.shape) > 3 else disp_real.shape
            x_coords = x_coords.reshape(coords_shape)
            y_coords = y_coords.reshape(coords_shape)
            z_coords = z_coords.reshape(coords_shape)


            disp_real_data = disp_real[..., :3] if len(disp_real.shape) > 3 else disp_real
            disp_imag_data = disp_imag[..., :3] if len(disp_imag.shape) > 3 else disp_imag

            data_samples.append({
                'sample_idx': i,
                't1': t1,
                't1_affine': t1_affine,
                'mask': mask,
                'mask_affine': mask_affine,
                'freq': freq,
                'direction': direction_value,
                'direction_label': direction,
                'sex': sex_value,
                'sex_label': sex,
                'age': age,
                'brain_volume': brain_volume,
                'x_coords': x_coords, 
                'y_coords': y_coords,
                'z_coords': z_coords, #80x80x44
                'disp_real': disp_real_data,
                'disp_imag': disp_imag_data,
                'disp_re_affine': disp_re_affine,
                'disp_im_affine': disp_im_affine,
                'shape': {
                    't1': t1.shape,
                    'disp': disp_real_data.shape,
                    'mask': mask.shape,
                }
            })
    return data_samples


def voxel_normalize(f,samples):
    sample_shape = f[samples[0]]['t1'].shape  
    voxel_min = np.full(sample_shape, np.inf) 
    voxel_max = np.full(sample_shape, -np.inf) 

    for sample_name in samples:
        t1_image = f[sample_name]['t1'][:] 
        
        voxel_min = np.minimum(voxel_min, t1_image)
        voxel_max = np.maximum(voxel_max, t1_image)
    return voxel_min, voxel_max

def compute_scaling_factors(file_path):
    scaling_factors = {}
    demo_stats = {'age_min': float('inf'), 'age_max': -float('inf'),
                  'brain_volume_min': float('inf'), 'brain_volume_max': -float('inf')}

    disp_global_min = float('inf')
    disp_global_max = -float('inf')

    with h5py.File(file_path, 'r') as f:
        samples = [k for k in f.keys() if k.startswith('sample_')]

        t1_min, t1_max = voxel_normalize(f, samples)

        for sample_name in samples:
            sample = f[sample_name]

            age = sample.attrs.get('age', -1)
            brain_volume = sample.attrs.get('brain_volume', -1)

            if age > 0:
                demo_stats['age_min'] = min(demo_stats['age_min'], age)
                demo_stats['age_max'] = max(demo_stats['age_max'], age)
            if brain_volume > 0:
                demo_stats['brain_volume_min'] = min(demo_stats['brain_volume_min'], brain_volume)
                demo_stats['brain_volume_max'] = max(demo_stats['brain_volume_max'], brain_volume)

            disp_real = sample['displacement/real/data'][:].astype(np.float32)
            disp_mag = np.linalg.norm(disp_real[..., :3], axis=-1)  # magnitude
            disp_global_min = min(disp_global_min, np.min(disp_mag))
            disp_global_max = max(disp_global_max, np.max(disp_mag))

        scaling_factors['demographics'] = demo_stats.copy()
        scaling_factors['disp_min'] = disp_global_min
        scaling_factors['disp_max'] = disp_global_max
        scaling_factors['t1_min'] = t1_min
        scaling_factors['t1_max'] = t1_max

    return scaling_factors

def scale_data_minmax(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

def unscale_data_minmax(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val + 1e-8) + min_val

def scale(data, subject_id, data_type, scaling_factors):
    if data_type == 'disp':
        min_val = scaling_factors['disp_min']
        max_val = scaling_factors['disp_max']
    elif data_type == 't1':
        min_val = scaling_factors['t1_min']
        max_val = scaling_factors['t1_max']
    else:
        factors = scaling_factors[f'sample_{subject_id}']
        min_val = factors[f'{data_type}_min']
        max_val = factors[f'{data_type}_max']
    return scale_data_minmax(data, min_val, max_val)


import torch
import torch.nn.functional as F
def scale_data(data, scaling_factors):
    batch_data = []
    for subject in data:
        sample_idx = subject['sample_idx']

        mask = subject['mask']
        freq_scaled = subject['freq'] / 100.0
        direction_value = subject['direction']
        sex_value = subject['sex']

        demo_stats = scaling_factors['demographics']
        age = subject['age']
        brain_volume = subject['brain_volume']

        age_normalized = (age - demo_stats['age_min']) / (demo_stats['age_max'] - demo_stats['age_min'] + 1e-8) if age > 0 else -1
        brain_volume_normalized = 0.1 + 0.8 * (brain_volume - demo_stats['brain_volume_min']) / (demo_stats['brain_volume_max'] - demo_stats['brain_volume_min'] + 1e-8) if brain_volume > 0 else -1

        t1_scaled = scale(subject['t1'], sample_idx, 't1', scaling_factors)
        t1_masked_scaled = t1_scaled * mask

        disp_real_scaled = scale(subject['disp_real'], sample_idx, 'disp', scaling_factors)
        disp_real_masked_scaled = disp_real_scaled * mask[..., None]

        t1_tensor = torch.tensor(t1_masked_scaled, dtype=torch.float32).unsqueeze(0) 

        t1_downsampled = F.interpolate(
            t1_tensor.unsqueeze(0), 
            size=(40, 40, 22),      
            mode='trilinear',
            align_corners=False
        ).squeeze(0) 

        disp_tensor = torch.tensor(disp_real_masked_scaled, dtype=torch.float32)  # [D, H, W, 3]
        disp_tensor = disp_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, D, H, W]

        disp_downsampled = F.interpolate(
            disp_tensor,
            size=(40, 40, 22),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # [3, 20, 20, 11]

        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        mask_downsampled = F.interpolate(
            mask_tensor, 
            size=(40, 40, 22), 
            mode='trilinear', 
            align_corners=False
        )
        mask_downsampled_binary = (mask_downsampled >= 0.5).float().squeeze(0).squeeze(0)

        batch_data.append({
            'sample_idx': sample_idx,
            't1': torch.tensor(t1_masked_scaled, dtype=torch.float32),
            't1_downsample': t1_downsampled.squeeze(0),
            't1_affine': torch.tensor(subject['t1_affine'], dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'mask_downsample': mask_downsampled_binary,
            'mask_affine': torch.tensor(subject['mask_affine'], dtype=torch.float32),
            'freq': torch.tensor(freq_scaled, dtype=torch.float32),
            'direction': torch.tensor(direction_value, dtype=torch.float32),
            'sex': torch.tensor(sex_value, dtype=torch.float32),
            'age': torch.tensor(age_normalized, dtype=torch.float32),
            'brain_volume': torch.tensor(brain_volume_normalized, dtype=torch.float32),
            'coords': {
                'x': torch.tensor(subject['x_coords'].reshape(-1), dtype=torch.float32),
                'y': torch.tensor(subject['y_coords'].reshape(-1), dtype=torch.float32),
                'z': torch.tensor(subject['z_coords'].reshape(-1), dtype=torch.float32)
            },
            'disp_real': torch.tensor(disp_tensor, dtype=torch.float32),
            'disp_real_downsample': disp_downsampled,
        })

    return batch_data


def combine_t1_with_scalar_data(t1_data, scalar_data):
    if t1_data.ndim == 3:
        t1_data = t1_data.unsqueeze(0)  # [1, D, H, W]
    
    def project_scalar(value):
        return torch.full_like(t1_data, float(value))  # [1, D, H, W]

    age = project_scalar(scalar_data.get('age', 0.0))
    brain_vol = project_scalar(scalar_data.get('brain_volume', 0.0))
    sex = project_scalar(scalar_data.get('sex', 0.0))
    freq = project_scalar(scalar_data.get('freq', 0.0))
    direction = project_scalar(scalar_data.get('direction', 0.0))

    scalar_channels = torch.cat([t1_data, age, brain_vol, sex, freq, direction], dim=0)  # [6, D, H, W]

    _, D, H, W = t1_data.shape
    z = torch.linspace(0, 1, steps=D, dtype=torch.float32, device=t1_data.device)
    y = torch.linspace(0, 1, steps=H, dtype=torch.float32, device=t1_data.device)
    x = torch.linspace(0, 1, steps=W, dtype=torch.float32, device=t1_data.device)
    
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # (D, H, W)
    
    pos_x = xx.unsqueeze(0)  # [1, D, H, W]
    pos_y = yy.unsqueeze(0)
    pos_z = zz.unsqueeze(0)

    positional_channels = torch.cat([pos_x, pos_y, pos_z], dim=0)  # [3, D, H, W]
    
    combined = torch.cat([scalar_channels, positional_channels], dim=0)  # [9, D, H, W]

    return combined


class FactorizedSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes2, 2))
        self.weights3 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes3, 2))

    def forward(self, x):
        out = 0

        x_ft = torch.fft.fft(x, dim=2)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes1] = self.complex_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        out += torch.fft.ifft(out_ft, dim=2).real

        x_ft = torch.fft.fft(x, dim=3)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :, :self.modes2] = self.complex_mul1d(x_ft[:, :, :, :self.modes2], self.weights2)
        out += torch.fft.ifft(out_ft, dim=3).real

        x_ft = torch.fft.fft(x, dim=4)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :, :, :self.modes3] = self.complex_mul1d(x_ft[:, :, :, :, :self.modes3], self.weights3)
        out += torch.fft.ifft(out_ft, dim=4).real

        return out

    def complex_mul1d(self, input, weights):

        weights_complex = torch.complex(weights[..., 0], weights[..., 1])
 
        batch_size = input.shape[0]
        spatial_dims = input.shape[2:]
        
        result = torch.zeros(batch_size, self.out_channels, *spatial_dims, dtype=input.dtype, device=input.device)
        
        for mode_idx in range(min(input.shape[-1], weights_complex.shape[2])):

            input_slice = input[..., mode_idx]
            weight_slice = weights_complex[:, :, mode_idx]
            
            result[..., mode_idx] = torch.einsum('bi...,io->bo...', input_slice, weight_slice)
        
        return result


class FFNOLayer(nn.Module):
    def __init__(self, width, modes1, modes2, modes3):
        super().__init__()
        self.spectral = FactorizedSpectralConv3d(width, width, modes1, modes2, modes3)
        self.w1 = nn.Conv3d(width, width * 2, kernel_size=1)
        self.w2 = nn.Conv3d(width * 2, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        z = self.spectral(x)
        z = self.w1(z)
        z = self.act(z)
        z = self.w2(z)
        return x + self.act(z)  

from torch.nn.utils import weight_norm

class FFNO3D(nn.Module):
    def __init__(self,
                 in_channels=9,        
                 out_channels=3,     
                 modes1=40, modes2=40, modes3=12,
                 width=64):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width

        self.lift = nn.Sequential(
            weight_norm(nn.Conv3d(in_channels, width, kernel_size=1)),
            nn.GELU()
        )

        self.norm = nn.InstanceNorm3d(width)

        self.layer0 = FFNOLayer(width, modes1, modes2, modes3)
        self.layer1 = FFNOLayer(width, modes1, modes2, modes3)
        self.layer2 = FFNOLayer(width, modes1, modes2, modes3)
        self.layer3 = FFNOLayer(width, modes1, modes2, modes3)

        self.output = nn.Sequential(
            weight_norm(nn.Conv3d(width, 128, kernel_size=1)),
            nn.GELU(),
            weight_norm(nn.Conv3d(128, out_channels, kernel_size=1))
        )

    def forward(self, x):
        x = self.lift(x)
        x = self.norm(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.output(x)
        return x

    
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['input'], sample['output'], sample['mask'] 

def create_dataloaders(train_samples, val_samples,test_samples, batch_size):
    train_dataset = MRIDataset(train_samples)
    val_dataset = MRIDataset(val_samples)
    test_dataset = MRIDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=500, learning_rate=1e-3, device='cuda', save_path='FFNO_Real.pth'):
    model = model.to(device)
    criterion = nn.MSELoss(reduction='none')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs * masks.unsqueeze(1)
            loss_per_voxel = criterion(outputs, targets)
            masked_loss = loss_per_voxel * masks.unsqueeze(1)
            total_loss = masked_loss.sum() / (masks.sum() * 3 + 1e-8)

            total_loss.backward()
            optimizer.step()


            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                outputs = model(inputs) * masks.unsqueeze(1)
                loss = criterion(outputs, targets) * masks.unsqueeze(1)
                val_batch_loss = loss.sum() / (masks.sum() * 3 + 1e-8)
                val_loss += val_batch_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        #print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("FFNO_Real_Loss.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets, masks in tqdm(test_loader, desc="Evaluating"):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            outputs = model(inputs)

            if outputs.shape != targets.shape:
                targets = targets.permute(0, 4, 1, 2, 3).contiguous()

            masked_outputs = outputs * masks.unsqueeze(1)
            masked_targets = targets * masks.unsqueeze(1)

            loss_per_voxel = criterion(masked_outputs, masked_targets)
            loss_sum = loss_per_voxel.sum()

            mask_sum = masks.sum() * outputs.shape[1]  

            total_loss += loss_sum / (mask_sum.item() + 1e-8)  

    avg_loss = total_loss / len(test_loader)
    rmse = avg_loss.sqrt()
    print(f"Test Loss: {avg_loss:.6f} | RMSE: {rmse:.6f}")
    return avg_loss.item(), rmse.item()


import os
import torch
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('./TBI_Results', exist_ok=True)

model = FFNO3D(
    in_channels=9,  # T1 + 5 scalar features 
    out_channels=3,  # XYZ displacement components
    modes1=40, 
    modes2=40, 
    modes3=12, 
    width=96
)
total_params = sum(p.numel() for p in model.parameters())

file_path = './' #update with file path
train_ratio = 0.7
data_samples = load_and_preprocess_data(file_path)
scaling_factors = compute_scaling_factors(file_path)
scaled_data = scale_data(data_samples, scaling_factors)
processed_samples = []
for i, sample in enumerate(scaled_data):

    scalar_data = {
        'age': sample['age'],
        'sex': sample['sex'],
        'brain_volume': sample['brain_volume'],
        'freq': sample['freq'],
        'direction': sample['direction']
    }

    t1_data = torch.tensor(sample['t1'], dtype=torch.float32)  # [1, D, H, W]

    combined_input = combine_t1_with_scalar_data(t1_data, scalar_data)

    output_tensor = torch.tensor(sample['disp_real'], dtype=torch.float32)  # [3, D, H, W]

    processed_samples.append({
    'input': combined_input,  # [6, D, H, W]
    'output': output_tensor,  # [3, D, H, W]
    'mask': sample['mask']  # [D, H, W]
})


val_ratio=0.1
n_samples = len(processed_samples)
n_train = int(n_samples * train_ratio)
n_val = int(n_samples * val_ratio)


np.random.seed(42)
np.random.shuffle(processed_samples)
train_samples = processed_samples[:n_train]
val_samples = processed_samples[n_train:n_train + n_val]
test_samples = processed_samples[n_train + n_val:]
    
train_loader, val_loader, test_loader = create_dataloaders(train_samples, val_samples, test_samples, batch_size)

save_path = os.path.join('./', 'FFNO_Real.pth')
train_losses, val_losses = train_model(
    model=model, 
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=500,
    learning_rate=lr,
    device=device,
    save_path=save_path
)


model.load_state_dict(torch.load(save_path)['model_state_dict'])
avg_loss, rmse = evaluate_model(model, test_loader, device)

with open(os.path.join('./', 'FFNO_Real_test_metrics.txt'), 'w') as f:
    f.write(f"Average Loss: {avg_loss:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")

