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
epochs = 1000
vis_iter = 1000
lr = 1e-3
transition_steps = 2000
decay_rate = 0.9

t1_small_img = torch.ones((1, 40, 40, 22)) 
scalar_input = torch.ones((1, 1))          
freq_in = torch.ones((1, 1))               
t1_small_img_trunk = torch.ones((1, 40*40*22, trunk_input_features))  
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


        disp_tensor = torch.tensor(disp_real_masked_scaled, dtype=torch.float32)  # [D, H, W, 3]
        disp_tensor = disp_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, D, H, W]

        batch_data.append({
            'sample_idx': sample_idx,
            't1': torch.tensor(t1_masked_scaled, dtype=torch.float32),
            't1_affine': torch.tensor(subject['t1_affine'], dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
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

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2) * self.scale)

    
    def complex_mul3d(self, input, weights):
        real = torch.einsum("bixyz,ioxyz->boxyz", input[..., 0], weights[..., 0]) - \
               torch.einsum("bixyz,ioxyz->boxyz", input[..., 1], weights[..., 1])
        imag = torch.einsum("bixyz,ioxyz->boxyz", input[..., 0], weights[..., 1]) + \
               torch.einsum("bixyz,ioxyz->boxyz", input[..., 1], weights[..., 0])
        return torch.stack([real, imag], dim=-1)
    
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-3), x.size(-2), x.size(-1)//2 + 1, 2,
            device=x.device, dtype=torch.float
        )
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.complex_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
            self.weights
        )
        
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=[-3, -2, -1])

        return x
    
from torch.nn.utils import weight_norm
class FNO3DImproved(nn.Module):
    def __init__(self, 
                 in_channels=9,       # T1 + 5 scalar
                 out_channels=3,      # XYZ displacement  
                 modes1=22, modes2=40, modes3=21, 
                 width=64):
        super(FNO3DImproved, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        
        
        self.lift = nn.Sequential(
            weight_norm(nn.Conv3d(in_channels, width, kernel_size=1)),
            nn.GELU()
        )

        self.norm = nn.InstanceNorm3d(width)

        self.dropout = nn.Dropout3d(p=0.1)
        
        self.conv0 = SpectralConv3d(width, width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(width, width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(width, width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(width, width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(width, width, kernel_size=1)
        self.w1 = nn.Conv3d(width, width, kernel_size=1)
        self.w2 = nn.Conv3d(width, width, kernel_size=1)
        self.w3 = nn.Conv3d(width, width, kernel_size=1)
        
        self.output = nn.Sequential(
            weight_norm(nn.Conv3d(width, 128, kernel_size=1)),
            nn.GELU(),
            weight_norm(nn.Conv3d(128, out_channels, kernel_size=1))
        )  
        
    def forward(self, x):
        x = self.lift(x)
        x = self.norm(x) 
        
        x1 = self.conv0(x)
        x1 = x1 + self.w0(x)
        x = F.gelu(x1)
        
        x1 = self.conv1(x)
        x1 = x1 + self.w1(x)
        x = F.gelu(x1)
        
        x1 = self.conv2(x)
        x1 = x1 + self.w2(x)
        x = F.gelu(x1)
        
        x1 = self.conv3(x)
        x1 = x1 + self.w3(x)
        x = F.gelu(x1)
        
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('./TBI_Resuts', exist_ok=True)

model = FNO3DImproved(
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
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import os

save_dir = "./FNO_Visualization"
os.makedirs(save_dir, exist_ok=True)

def to_nifti(vol):
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    return nib.Nifti1Image(vol.numpy(), affine)

def pad_to_square(img):
    h, w = img.shape
    max_dim = max(h, w)
    
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    pad_h_extra = (max_dim - h) % 2
    pad_w_extra = (max_dim - w) % 2
    
    padded_img = np.pad(img, 
                       ((pad_h, pad_h + pad_h_extra), 
                        (pad_w, pad_w + pad_w_extra)), 
                       mode='constant', constant_values=img.min())
    
    return padded_img

def get_center_slice(volume):
    d, h, w = volume.shape
    axial = volume[d//2, :, :]      # axial (transverse)
    coronal = volume[:, h//2, :]    # coronal (frontal)
    sagittal = volume[:, :, w//2]   # sagittal (lateral)
    
    axial_padded = pad_to_square(axial)
    coronal_padded = pad_to_square(coronal)
    sagittal_padded = pad_to_square(sagittal)
    
    return axial_padded, coronal_padded, sagittal_padded

def plot_panel(imgs, titles, cmaps, vmins, vmaxs, fig_title, save_path):
    n_rows = len(imgs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3 * n_rows), constrained_layout=True)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    orientations = ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)']

    for i in range(n_rows):
        slices = get_center_slice(imgs[i])
        im = None  

        for j in range(3):
            ax = axes[i, j]
            im = ax.imshow(np.rot90(slices[j]), cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
            ax.set_title(f"{titles[i]} - {orientations[j]}", fontsize=10)
            ax.axis('off')
            ax.set_aspect('equal')

        cbar = fig.colorbar(im, ax=axes[i, :], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        if i == 0: 
            cbar.ax.set_visible(False)

    fig.suptitle(fig_title, fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


checkpoint = torch.load("./FNO_Real.pth", map_location='cpu')

model.load_state_dict(checkpoint["model_state_dict"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for batch in test_loader:
        inputs, targets, masks = batch
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        preds = model(inputs)
        preds = preds * masks.unsqueeze(1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_masks.append(masks.cpu())

# Concatenate predictions, targets, and masks
all_preds = torch.cat(all_preds, dim=0).cpu()        # [N, C, D, H, W]
all_targets = torch.cat(all_targets, dim=0).cpu()
all_masks = torch.cat(all_masks, dim=0).cpu()

for i in range(len(test_loader)):
    input_sample, target_sample, mask_sample = test_loader.dataset[i]
    pred_disp = all_preds[i]
    target_disp = all_targets[i].squeeze(0)
    mask = all_masks[i]
    
    pred_mag = torch.norm(pred_disp, dim=0) * mask
    gt_mag = torch.norm(target_disp, dim=0) * mask
    t1_img = input_sample[0]
    
    # Calculate error (absolute difference)
    error_mag = torch.abs(pred_mag - gt_mag) * mask
    
    vmin = min(gt_mag.min().item(), pred_mag.min().item())
    vmax = max(gt_mag.max().item(), pred_mag.max().item())
    error_vmax = error_mag.max().item()
    
    imgs = [t1_img, gt_mag, pred_mag, error_mag]
    titles = ["T1 Input", "Ground Truth Displacement", "Predicted Displacement", "Absolute Error"]
    cmaps = ['gray', 'plasma', 'plasma', 'hot']
    vmins = [None, vmin, vmin, 0]
    vmaxs = [None, vmax, vmax, error_vmax]
    
    save_path = os.path.join(save_dir, f"sample_{i+1:02d}.png")
    plot_panel(imgs, titles, cmaps, vmins, vmaxs, f"Sample {i+1}", save_path)