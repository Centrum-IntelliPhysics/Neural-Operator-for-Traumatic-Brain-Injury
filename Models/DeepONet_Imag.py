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
trunk_input_features = 3
hidden_dim = 100
p_test = 100
result_dir = './'
epochs = 5000
vis_iter = 1000
lr = 1e-3
transition_steps = 2000
decay_rate = 0.9

t1_small_img = torch.ones((1, 80, 80, 44))  
scalar_input = torch.ones((1, 1))      
freq_in = torch.ones((1, 1))               
t1_small_img_trunk = torch.ones((1, 80*80*44, trunk_input_features)) 

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
    shape_stats = {}
    direction_stats = {'AP': 0, 'LR': 0}
    demographic_stats = {
        'sex': {'M': 0, 'F': 0, 'unknown': 0},
        'age': {'min': float('inf'), 'max': -float('inf'), 'mean': 0},
        'brain_volume': {'min': float('inf'), 'max': -float('inf'), 'mean': 0}
    }

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
            direction_stats[direction] += 1

            sex = sample.attrs.get('sex', 'unknown')
            age = sample.attrs.get('age', -1)
            brain_volume = sample.attrs.get('brain_volume', -1)

            # Update demographic statistics
            demographic_stats['sex'][sex] += 1
            if age > 0:
                demographic_stats['age']['min'] = min(demographic_stats['age']['min'], age)
                demographic_stats['age']['max'] = max(demographic_stats['age']['max'], age)
                total_age += age
            if brain_volume > 0:
                demographic_stats['brain_volume']['min'] = min(demographic_stats['brain_volume']['min'], brain_volume)
                demographic_stats['brain_volume']['max'] = max(demographic_stats['brain_volume']['max'], brain_volume)
                total_brain_volume += brain_volume

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

            # Process displacement data
            disp_real_data = disp_real[..., :3] if len(disp_real.shape) > 3 else disp_real
            disp_imag_data = disp_imag[..., :3] if len(disp_imag.shape) > 3 else disp_imag

            data_samples.append({
                'subject_idx': i,
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
                'z_coords': z_coords,
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

            shape_key = (t1.shape, disp_real_data.shape)
            if shape_key not in shape_stats:
                shape_stats[shape_key] = 0
            shape_stats[shape_key] += 1

        demographic_stats['age']['mean'] = total_age / n_samples if n_samples > 0 else 0
        demographic_stats['brain_volume']['mean'] = total_brain_volume / n_samples if n_samples > 0 else 0
    return data_samples

def voxel_normalize(f, samples):
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

    # global demographic stats
    demo_stats = {
        'age_min': float('inf'), 'age_max': -float('inf'),
        'brain_volume_min': float('inf'), 'brain_volume_max': -float('inf')
    }

    disp_global_min = float('inf')
    disp_global_max = -float('inf')

    with h5py.File(file_path, 'r') as f:
        samples = [k for k in f.keys() if k.startswith('sample_')]

        t1_min, t1_max = voxel_normalize(f, samples)

        for sample_name in samples:
            sample = f[sample_name]

            # demographics
            age = sample.attrs.get('age', -1)
            brain_volume = sample.attrs.get('brain_volume', -1)

            if age > 0:
                demo_stats['age_min'] = min(demo_stats['age_min'], age)
                demo_stats['age_max'] = max(demo_stats['age_max'], age)
            if brain_volume > 0:
                demo_stats['brain_volume_min'] = min(demo_stats['brain_volume_min'], brain_volume)
                demo_stats['brain_volume_max'] = max(demo_stats['brain_volume_max'], brain_volume)

            disp_imag = sample['displacement/imag/data'][:].astype(np.float32)
            disp_mag = np.linalg.norm(disp_imag[..., :3], axis=-1)
            disp_global_min = min(disp_global_min, np.min(disp_mag))
            disp_global_max = max(disp_global_max, np.max(disp_mag))

        scaling_factors['demographics'] = demo_stats.copy()
        scaling_factors['disp_min'] = disp_global_min
        scaling_factors['disp_max'] = disp_global_max
        scaling_factors['t1_min'] = t1_min
        scaling_factors['t1_max'] = t1_max

    return scaling_factors


import torch

def scale_data_minmax(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

def unscale_data_minmax(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val + 1e-8) + min_val

def scale_data_helper(data, data_type, scaling_factors):
    if data_type == 't1':
        min_val = scaling_factors['t1_min']
        max_val = scaling_factors['t1_max']
    elif data_type == 'disp':
        min_val = scaling_factors['disp_min']
        max_val = scaling_factors['disp_max']
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    return scale_data_minmax(data, min_val, max_val)

def scale_data(data, scaling_factors):
    batch_data = []
    for sample in data:
        mask = sample['mask']
        subject_idx = sample['subject_idx']

        t1_scaled = scale_data_helper(sample['t1'], 't1', scaling_factors)
        t1_masked_scaled = t1_scaled * mask

        disp_real_scaled = scale_data_helper(sample['disp_real'], 'disp', scaling_factors)
        disp_real_masked_scaled = disp_real_scaled * mask[..., None]

        disp_imag_scaled = scale_data_helper(sample['disp_imag'], 'disp', scaling_factors)
        disp_imag_masked_scaled = disp_imag_scaled * mask[..., None]

        demo_stats = scaling_factors['demographics']
        age_normalized = (sample['age'] - demo_stats['age_min']) / (demo_stats['age_max'] - demo_stats['age_min'] + 1e-8) if sample['age'] > 0 else -1
        brain_volume_normalized = 0.1 + 0.8 * (sample['brain_volume'] - demo_stats['brain_volume_min']) / (demo_stats['brain_volume_max'] - demo_stats['brain_volume_min'] + 1e-8) if sample['brain_volume'] > 0 else -1

        batch_data.append({
            'inputs': {
                'sample_idx': subject_idx,
                't1': torch.tensor(t1_masked_scaled, dtype=torch.float32),
                't1_affine': torch.tensor(sample['t1_affine'], dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'mask_affine': torch.tensor(sample['mask_affine'], dtype=torch.float32),
                'freq': torch.tensor(sample['freq']/100.0, dtype=torch.float32),
                'direction': torch.tensor(sample['direction'], dtype=torch.float32),
                'sex': torch.tensor(sample['sex'], dtype=torch.float32),
                'age': torch.tensor(age_normalized, dtype=torch.float32),
                'brain_volume': torch.tensor(brain_volume_normalized, dtype=torch.float32),
                'coords': {
                    'x': torch.tensor(sample['x_coords'].reshape(-1), dtype=torch.float32),
                    'y': torch.tensor(sample['y_coords'].reshape(-1), dtype=torch.float32),
                    'z': torch.tensor(sample['z_coords'].reshape(-1), dtype=torch.float32)
                },
            },
            'output': {
                'real': {
                    'data': torch.tensor(disp_real_masked_scaled, dtype=torch.float32),
                    'affine': torch.tensor(sample['disp_re_affine'], dtype=torch.float32)
                },
                'imag': {
                    'data': torch.tensor(disp_imag_masked_scaled, dtype=torch.float32),
                    'affine': torch.tensor(sample['disp_im_affine'], dtype=torch.float32)
                }
            },
            'scaling_values': {
                't1_min': torch.tensor(scaling_factors['t1_min']),
                't1_max': torch.tensor(scaling_factors['t1_max']),
                'disp_min': torch.tensor(scaling_factors['disp_min']),
                'disp_max': torch.tensor(scaling_factors['disp_max'])
            }
        })
    return batch_data

class CNNBranchNetwork(nn.Module):
    def __init__(self, features, output_dim=300):
        super().__init__()
        self.features = features
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(44, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(2, 2)
        
        self.flatten_size = 64 * 10 * 10 
        
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, self.output_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class FNNBranchNetwork(nn.Module):
    def __init__(self, fnn_layers, output_dim=300):  
        super().__init__()
        self.output_dim = output_dim
        
        layers = []
        in_dim = 1 
        for fs in fnn_layers[:-1]:
            layers.append(nn.Linear(in_dim, fs))
            layers.append(nn.ReLU())
            in_dim = fs
        layers.append(nn.Linear(in_dim, output_dim))  
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1) 
        return self.net(x)
    
class DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_layers, output_dim=3):
        super().__init__()
        self.output_dim = output_dim
        self.trunk_layers = trunk_layers
        
        branch_output_dim = 100 * output_dim  
        
        self.cnn_branch_net = CNNBranchNetwork(branch_features, output_dim=branch_output_dim)
        self.fnn_branch_net1 = FNNBranchNetwork(branch_features, output_dim=branch_output_dim)  # direction
        self.fnn_branch_net2 = FNNBranchNetwork(branch_features, output_dim=branch_output_dim)  # frequency
        self.fnn_branch_net3 = FNNBranchNetwork(branch_features, output_dim=branch_output_dim)  # sex
        self.fnn_branch_net4 = FNNBranchNetwork(branch_features, output_dim=branch_output_dim)  # brain volume
        self.fnn_branch_net5 = FNNBranchNetwork(branch_features, output_dim=branch_output_dim)  # age

        layers = []
        in_dim = 3 
        for i, fs in enumerate(trunk_layers[:-1]):
            layers.append(nn.Linear(in_dim, fs * output_dim))
            layers.append(nn.ReLU())
            in_dim = fs * output_dim
        layers.append(nn.Linear(in_dim, trunk_layers[-1] * output_dim))
        self.trunk_net = nn.Sequential(*layers)

        self.output_bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, t1_image, scalar_input, freq, sex, bv, age, trunk_input):
        cnn_branch_x = self.cnn_branch_net(t1_image)      # [B, 300]
        fnn_branch_x1 = self.fnn_branch_net1(scalar_input)  # [B, 300]
        fnn_branch_x2 = self.fnn_branch_net2(freq)          # [B, 300]
        fnn_branch_x3 = self.fnn_branch_net3(sex)           # [B, 300]
        fnn_branch_x4 = self.fnn_branch_net4(bv)            # [B, 300]
        fnn_branch_x5 = self.fnn_branch_net5(age)           # [B, 300]

        branch_x = cnn_branch_x * fnn_branch_x1 * fnn_branch_x2 * fnn_branch_x3 * fnn_branch_x4 * fnn_branch_x5  # [B, 300]

        B, N, D = trunk_input.shape
        trunk_x = self.trunk_net(trunk_input)  

        result_x = torch.einsum('bi,bji->bj', branch_x[:, 0:100], trunk_x[:, :, 0:100])
        result_y = torch.einsum('bi,bji->bj', branch_x[:, 100:200], trunk_x[:, :, 100:200])
        result_z = torch.einsum('bi,bji->bj', branch_x[:, 200:300], trunk_x[:, :, 200:300])

        result = torch.stack([result_x, result_y, result_z], dim=-1)  # [B, N, 3]
        result = result + self.output_bias

        return result
    
def train_step(model, optimizer, batch_data, device='cuda'):

    model.train()
    optimizer.zero_grad()

    
    t1_image, scalar_input, freq, sex, bv, age, trunk_input, targets = batch_data
    t1_image = t1_image.to(device)
    scalar_input = scalar_input.to(device)
    freq = freq.to(device)
    sex = sex.to(device)
    bv = bv.to(device)
    age = age.to(device)
    trunk_input = trunk_input.to(device)
    targets = targets.to(device)

    outputs = model(t1_image, scalar_input, freq, sex, bv, age, trunk_input)  # [B, N, 3]

    loss_fn = nn.MSELoss()
    loss = loss_fn(outputs, targets)

    loss.backward()
    optimizer.step()

    metrics = {
        'loss': loss.item(),
        'loss_x': nn.MSELoss()(outputs[..., 0], targets[..., 0]).item(),
        'loss_y': nn.MSELoss()(outputs[..., 1], targets[..., 1]).item(),
        'loss_z': nn.MSELoss()(outputs[..., 2], targets[..., 2]).item()
    }

    return loss.item(), metrics

import torch

def prepare_batch_input(batch_data, device='cuda'):

    t1_images = torch.stack([
        torch.tensor(sample['inputs']['t1'], dtype=torch.float32)
        for sample in batch_data
    ]).to(device)

    coords_batch = []
    target_imag_batch = []
    masks_batch = []
    freq_batch = []
    direction_batch = []
    sex_batch = []
    bv_batch = []
    age_batch = []

    for sample in batch_data:
        coords = torch.column_stack([
            torch.tensor(sample['inputs']['coords']['x'], dtype=torch.float32),
            torch.tensor(sample['inputs']['coords']['y'], dtype=torch.float32),
            torch.tensor(sample['inputs']['coords']['z'], dtype=torch.float32)
        ])
        coords_batch.append(coords)

        direction_batch.append(torch.tensor(sample['inputs']['direction'], dtype=torch.float32))
        freq_batch.append(torch.tensor(sample['inputs']['freq'], dtype=torch.float32))
        sex_batch.append(torch.tensor(sample['inputs']['sex'], dtype=torch.float32))
        bv_batch.append(torch.tensor(sample['inputs']['brain_volume'], dtype=torch.float32))
        age_batch.append(torch.tensor(sample['inputs']['age'], dtype=torch.float32))

        mask = torch.tensor(sample['inputs']['mask'], dtype=torch.float32)
        masks_batch.append(mask)

        target_imag = torch.tensor(sample['output']['imag']['data'], dtype=torch.float32).reshape(-1, 3)
        target_imag_batch.append(target_imag)

    trunk_input = torch.stack(coords_batch).to(device)
    target_imag = torch.stack(target_imag_batch).to(device)
    masks = torch.stack(masks_batch).to(device)
    directions = torch.stack(direction_batch).reshape(-1, 1).to(device)
    freqs = torch.stack(freq_batch).reshape(-1, 1).to(device)
    sexs = torch.stack(sex_batch).reshape(-1, 1).to(device)
    BVs = torch.stack(bv_batch).reshape(-1, 1).to(device)
    ages = torch.stack(age_batch).reshape(-1, 1).to(device)


    return t1_images, trunk_input, target_imag, masks, directions, freqs, sexs, BVs, ages

import torch
import torch.nn.functional as F

def loss_fn(model, batch_data, device='cuda'):

    t1_images, trunk_input, target_imag, masks, directions, freqs, sexs, BVs, ages = prepare_batch_input(batch_data, device=device)

    pred_imag = model(t1_images, directions, freqs, sexs, BVs, ages, trunk_input)  # [B, N, 3]

    masks = masks.reshape(masks.shape[0], -1, 1)  # [B, N, 1]
    masked_pred_imag = pred_imag * masks

    batch_mse = torch.mean((target_imag - masked_pred_imag) ** 2, dim=(1, 2))  # [B]
    loss_imag = torch.mean(batch_mse) 

    total_loss = loss_imag

    metrics = {
        'loss_imag': loss_imag.item(),
        'total_loss': total_loss.item()
    }

    return total_loss, metrics



class MRIDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        inputs = sample['input']
        output = sample['output']
        
        t1_image = inputs['t1']
        mask = inputs['mask']
        direction = inputs['direction'].unsqueeze(0) if inputs['direction'].dim() == 0 else inputs['direction']
        freq = inputs['freq'].unsqueeze(0) if inputs['freq'].dim() == 0 else inputs['freq']
        sex = inputs['sex'].unsqueeze(0) if inputs['sex'].dim() == 0 else inputs['sex']
        brain_volume = inputs['brain_volume'].unsqueeze(0) if inputs['brain_volume'].dim() == 0 else inputs['brain_volume']
        age = inputs['age'].unsqueeze(0) if inputs['age'].dim() == 0 else inputs['age']
        
        coords = torch.column_stack([
            inputs['coords']['x'],
            inputs['coords']['y'],
            inputs['coords']['z']
        ])
        
        target = output['imag']['data'].reshape(-1, 3)
        
        return (t1_image, target, mask, direction, freq, sex, brain_volume, age, coords)

def create_dataloaders(train_samples, val_samples,test_samples, batch_size):
    train_dataset = MRIDataset(train_samples)
    val_dataset = MRIDataset(val_samples)
    test_dataset = MRIDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_deeponet(model, train_loader, val_loader, num_epochs=5000,
                   learning_rate=1e-3, device='cuda', save_dir='./',
                   log_file='DeepONet_Imag.csv', save_model_name='DeepONet_Imag.pth'):
    model = model.to(device)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_file)

    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    best_val_loss = float('inf')
    train_losses, val_losses = [], []


    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            t1_images, targets, masks, directions, freqs, sexs, bvs, ages, trunk_input = batch_data
            t1_images = t1_images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            directions = directions.to(device)
            freqs = freqs.to(device)
            sexs = sexs.to(device)
            bvs = bvs.to(device)
            ages = ages.to(device)
            trunk_input = trunk_input.to(device)

            optimizer.zero_grad()
            outputs = model(t1_images, directions, freqs, sexs, bvs, ages, trunk_input)

            B, H, W, D = masks.shape
            masks_flat = masks.reshape(B, -1)  # [B, N] where N = H*W*D
            masks_exp = masks_flat.unsqueeze(-1)  # [B, N, 1]

            targets_flat = targets.reshape(B, -1, 3)  # [B, N, 3]

            loss_per_voxel = criterion(outputs, targets_flat)
            masked_loss = loss_per_voxel * masks_exp
            total_loss = masked_loss.sum() / (masks_flat.sum() * outputs.shape[-1] + 1e-8)

            total_loss.backward()
            optimizer.step()

            train_loss_epoch += total_loss.item()

        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)

        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                t1_images, targets, masks, directions, freqs, sexs, bvs, ages, trunk_input = batch_data
                t1_images = t1_images.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                directions = directions.to(device)
                freqs = freqs.to(device)
                sexs = sexs.to(device)
                bvs = bvs.to(device)
                ages = ages.to(device)
                trunk_input = trunk_input.to(device)

                outputs = model(t1_images, directions, freqs, sexs, bvs, ages, trunk_input)

                B, H, W, D = masks.shape
                masks_flat = masks.reshape(B, -1)  # [B, N]
                masks_exp = masks_flat.unsqueeze(-1)  # [B, N, 1]
                targets_flat = targets.reshape(B, -1, 3)  # [B, N, 3]

                loss = criterion(outputs, targets_flat) * masks_exp
                val_batch_loss = loss.sum() / (masks_flat.sum() * outputs.shape[-1] + 1e-8)
                val_loss_epoch += val_batch_loss.item()

        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)
        scheduler.step(val_loss_epoch)

        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{train_loss_epoch:.6e},{val_loss_epoch:.6e}\n')

        #print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_epoch:.6f} | Val Loss: {val_loss_epoch:.6f}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            save_path = os.path.join(save_dir, save_model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            #print(f"New best model saved at {save_path} with val loss {best_val_loss:.6f}")

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "DeepONet_Imag_Loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    #print(f"Training complete. Loss plot saved in {save_dir}")

    return train_losses, val_losses


import os
import torch
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('./TBI_Results', exist_ok=True)
file_path = './' #update with file path

scaling_factors = compute_scaling_factors(file_path)
data_samples = load_and_preprocess_data(file_path)
scaled_data = scale_data(data_samples, scaling_factors)
processed_samples=[]
for i, sample in enumerate(scaled_data):
    processed_samples.append({
    'input': sample['inputs'],  # [6, D, H, W]
    'output': sample['output'],  # [3, D, H, W]
    'mask': sample['inputs']['mask']  # [D, H, W]
})

train_ratio = 0.7
val_ratio=0.1

n_samples = len(processed_samples)
n_train = int(n_samples * train_ratio)
n_val = int(n_samples * val_ratio)

np.random.seed(42)
np.random.shuffle(processed_samples)
train_samples = processed_samples[:n_train]
val_samples = processed_samples[n_train:n_train + n_val]
test_samples = processed_samples[n_train + n_val:]

train_loader, val_loader, test_loader = create_dataloaders(
    train_samples, val_samples, test_samples, batch_size
)

trunk_layers = [trunk_layers] if isinstance(trunk_layers, int) else trunk_layers
branch_layers = [branch_layers] if isinstance(branch_layers, int) else branch_layers

trunk_layers = trunk_layers + [hidden_dim]
branch_layers = branch_layers + [hidden_dim]
print(trunk_layers)
print(branch_layers)

trunk_layers = tuple(trunk_layers)
branch_layers = tuple(branch_layers)
print(trunk_layers, branch_layers)


num_outputs = 3
branch_features = branch_layers  
output_dim = num_outputs 

model = DeepONet(branch_features=branch_features, 
                 trunk_layers=trunk_layers, 
                 output_dim=output_dim)

train_losses, val_losses = train_deeponet(
    model=model,
    train_loader=train_loader,  
    val_loader=val_loader,   
    num_epochs=epochs,
    learning_rate=lr,
    device=device,
    save_dir=result_dir
)