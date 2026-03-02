import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict
from math import exp
import re
import traceback
import omegaconf
from omegaconf import OmegaConf
import wandb
import logging
from sklearn.model_selection import KFold, StratifiedKFold
import sys
import io


# 配置管理
cfg = OmegaConf.create({
    "data": {
        "root_dirs": [r"E:\Unet\New_Fusion_ExportReg1", r"E:\Unet\New_Fusion_ExportReg3"],  # 新增：多个区域数据目录，都属于hetao类型
        "output_dir": "究极64UTLLd1gemini236",
        "modis_pattern": "MODIS_2024_2024-*_NDVI_resampled.tif",
        "landsat_pattern": "Landsat_NDVI_2024_ID_LC*.tif",
        "landcover_file": "landcover.tif",
        "block_size": 64,  # 保持为64
        "overlap": 16,
        "train_ratio": 0.7,
        "batch_size": 16,
        "gradient_accum_steps": 4,
        "valid_threshold": 0.9,  # 降低到0.85以增加有效块
        "window_days": 365,
        "max_modis_per_target": 3,
        "max_landsat_per_target": 2,
        "data_augment": True,
        "region": "hetao",  # 保持hetao，所有区域统一使用hetao参数
        "normalize_range": True,
        "k_folds": 5,  # 新增：K折交叉验证
        "spatial_split": True  # 新增：按区域分割
    },
    "training": {
        "epochs": 200,  # 增加到50，确保充分训练
        "lr": 1e-4,
        "patience": 20,
        "alpha": 0.8,
        "logistic_weight_start": 0.01,
        "logistic_weight_end": 0.1,
        "use_wandb": False,
        "label_smoothing": 0,
        "weight_decay": 1e-5,
        "use_amp": True  # 新增：AMP混合精度
    },
    "model": {
        "use_attention": True,
        "use_transformer": True,
        "use_landcover": True,
        "mc_dropout": False,
        "num_samples": 1,  # 增加到10，提高MC Dropout可靠性
        "embed_params": True,
        "sigma": 0.05,
        "use_multiscale": True,  # 新增：使用多尺度特征提取
        "adaptive_logistic": True  # 新增：自适应Logistic
    }
})

# 区域参数
REGION_PARAMS = {
    "hetao": {
        1: {"name": "建筑用地", "K1": 0.0, "r1": 0.0, "t_01": 0.0, "K2": 0.0, "r2": 0.0, "t_02": 0.0, "c": -0.7, "use_logistic": False, "use_double_logistic": False},
        2: {"name": "林草地", "K1": 0.17, "r1": 0.1, "t_01": 167.0, "K2": -0.16, "r2": 0.05, "t_02": 260.0, "c": 0.08, "use_logistic": True, "use_double_logistic": True},
        3: {"name": "荒地", "K1": 0.35, "r1": 0.07, "t_01": 155.0, "K2": -0.3, "r2": 0.15, "t_02": 281.0, "c": 0.09, "use_logistic": True, "use_double_logistic": True},
        4: {"name": "农田", "K1": 0.54, "r1": 0.06, "t_01": 170.0, "K2": -0.7, "r2": 0.02, "t_02": 271.0, "c": 0.17, "use_logistic": True, "use_double_logistic": True},
        5: {"name": "水体", "K1": 0.0, "r1": 0.0, "t_01": 0.0, "K2": 0.0, "r2": 0.0, "t_02": 0.0, "c": -0.7, "use_logistic": False, "use_double_logistic": False}
    }
}
LANDCOVER_CLASSES = REGION_PARAMS.get(cfg.data.region, REGION_PARAMS["hetao"])

# 新增：归一化/反归一化函数
def normalize_ndvi(data):
    """将NDVI [-1,1] 归一化到 [0,1]"""
    return (data + 1.0) / 2.0

def denormalize_ndvi(norm_data):
    """将 [0,1] 反归一化到 [-1,1]"""
    return norm_data * 2.0 - 1.0

# 工具函数
def parse_date_from_filename(path):
    fname = os.path.basename(path)
    m = re.search(r'(\d{4})(\d{2})(\d{2})', fname) or re.search(r'(\d{4})-(\d{2})-(\d{2})', fname)
    if m:
        year, month, day = map(int, m.groups())
        return np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", 'D')
    raise ValueError(f"无法从文件名解析日期: {path}")

def calculate_day_diff(date1, date2):
    d1 = np.datetime64(date1, 'D')
    d2 = np.datetime64(date2, 'D')
    return int((d2 - d1).astype('timedelta64[D]') / np.timedelta64(1, 'D'))

def get_month(date):
    return int(np.datetime64(date, 'M').astype(str).split('-')[1])

def get_doy(date: np.datetime64) -> int:
    dt = date.astype(datetime)
    return dt.timetuple().tm_yday

def compute_valid_ratio(path, ref_path=None):
    data, _, mask = load_raster(path, ref_path=ref_path)
    total_pixels = np.prod(mask.shape[1:])
    valid_pixels = np.sum(mask)
    return valid_pixels / total_pixels if total_pixels > 0 else 0

def filter_valid_images(paths, ref_path=None, threshold=0.95):
    valid_paths = []
    for path in tqdm(paths, desc="筛选有效影像"):
        ratio = compute_valid_ratio(path, ref_path)
        if ratio > threshold:
            valid_paths.append(path)
        else:
            print(f"排除无效影像 {path}: 有效比例 {ratio:.2%}")
    return valid_paths

def filter_by_window(paths, dates, target_date, window_days, max_images, exclude_target=False):
    dated_paths = [(abs(calculate_day_diff(d, target_date)), p, d) for p, d in zip(paths, dates)]
    dated_paths = [dp for dp in dated_paths if dp[0] <= window_days]
    if exclude_target:
        dated_paths = [dp for dp in dated_paths if dp[2] != target_date]
    dated_paths.sort(key=lambda x: x[0])
    if len(dated_paths) == 0:
        return [], []
    if max_images:
        if len(dated_paths) > max_images:
            dated_paths = dated_paths[:max_images]
        elif len(dated_paths) < max_images:
            last_dp = dated_paths[-1]
            dated_paths.extend([last_dp for _ in range(max_images - len(dated_paths))])
    filtered_paths = [dp[1] for dp in dated_paths]
    filtered_dates = [dp[2] for dp in dated_paths]
    return filtered_paths, filtered_dates

def load_raster(path, ref_path=None, clip_range=(-1, 1)):
    if path is None:
        return np.zeros((1, 64, 64), dtype=np.float32), None, np.zeros((1, 64, 64), dtype=np.uint8)
    
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
        nodata = src.nodata
    
    if ref_path is not None:
        with rasterio.open(ref_path) as ref:
            data_reproj = np.empty(shape=(data.shape[0], ref.height, ref.width), dtype=data.dtype)
            # 修改：如果路径包含 "landcover"，使用 nearest 重采样；否则使用 bilinear
            resampling_method = Resampling.nearest if "landcover" in path.lower() else Resampling.bilinear
            for i in range(data.shape[0]):
                reproject(
                    source=data[i],
                    destination=data_reproj[i],
                    src_transform=profile['transform'],
                    src_crs=profile['crs'],
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=resampling_method
                )
            data = data_reproj
            profile = ref.profile.copy()
    
    mask = np.ones_like(data, dtype=np.uint8)
    for i in range(data.shape[0]):
        band = data[i].astype(np.float32)
        invalid = (np.isnan(band) | np.isinf(band) | (band < clip_range[0]) | (band > clip_range[1]))
        if nodata is not None:
            invalid = invalid | (band == nodata)
        band[invalid] = 0
        mask[i] = (~invalid).astype(np.uint8)
        data[i] = band
    
    return data, profile, mask

def save_raster(data, profile, output_path):
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)

# 双 Logistic NDVI 模型
def logistic_ndvi(t, K1, r1, t_01, K2, r2, t_02, c, use_double_logistic=True):
    if not use_double_logistic:
        return c
    return c + K1 / (1 + np.exp(-r1 * (t - t_01))) + K2 / (1 + np.exp(-r2 * (t - t_02)))

# 辅助模块
class SSIMLoss(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(g_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.W_x = nn.Sequential(nn.Conv2d(x_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class LandcoverGuidedAttention(nn.Module):
    def __init__(self, in_channels, lc_channels, inter_channels):
        super().__init__()
        self.conv_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))
        self.conv_lc = nn.Sequential(nn.Conv2d(lc_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))
        self.attention = nn.Sequential(nn.Conv2d(inter_channels * 2, inter_channels, 1), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, 1, 1), nn.Sigmoid())

    def forward(self, x, lc):
        if lc.size(2) != x.size(2) or lc.size(3) != x.size(3):
            lc = F.interpolate(lc, size=x.size()[2:], mode='bilinear', align_corners=False)
        x_feat = self.conv_x(x)
        lc_feat = self.conv_lc(lc)
        combined = torch.cat([x_feat, lc_feat], dim=1)
        attn = self.attention(combined)
        return x * attn

class TemporalAndLandcoverGuidedAttention(nn.Module):
    def __init__(self, in_channels, lc_channels, time_channels, inter_channels):
        super().__init__()
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_lc = nn.Sequential(
            nn.Conv2d(lc_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_t = nn.Sequential(
            nn.Conv2d(time_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(inter_channels * 3, inter_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lc, time_embed_broadcasted):
        if lc.size(2) != x.size(2) or lc.size(3) != x.size(3):
            lc = F.interpolate(lc, size=x.size()[2:], mode='bilinear', align_corners=False)
        if time_embed_broadcasted.size(2) != x.size(2) or time_embed_broadcasted.size(3) != x.size(3):
            time_embed_broadcasted = F.interpolate(time_embed_broadcasted, size=x.size()[2:], mode='bilinear', align_corners=False)
        x_feat = self.conv_x(x)
        lc_feat = self.conv_lc(lc)
        t_feat = self.conv_t(time_embed_broadcasted)
        combined = torch.cat([x_feat, lc_feat, t_feat], dim=1)
        attn = self.attention(combined)
        return x * attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerTimeEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.projection = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x.mean(dim=1)

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器（从模型1融合）"""
    def __init__(self, in_channels, out_channels: int):
        super().__init__()
        # 多尺度卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.dilated_conv = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=2, dilation=2)
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f_dilated = self.dilated_conv(x)
        combined = torch.cat([f1, f3, f5, f_dilated], dim=1)
        return self.fusion(combined)

class AdaptiveLogisticPredictor(nn.Module):
    """自适应Logistic参数预测器（从模型1融合）"""
    def __init__(self, in_channels: int, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        # 参数预测
        self.param_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, num_classes * 7)  # 7个参数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        params = self.param_predictor(features)
        params = params.view(-1, self.num_classes, 7)  # [B, num_classes, 7]
        return params

class GradientLoss(nn.Module):
    """梯度损失 - 支持多通道输入，对每个通道独立计算梯度后平均"""
    def __init__(self):
        super().__init__()
        # Sobel 滤波器 - 扩展为可处理多通道
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        if self.sobel_x.device != device:
            self.sobel_x = self.sobel_x.to(device)
            self.sobel_y = self.sobel_y.to(device)

        # 如果是单通道，保持原样；如果是多通道，对每个通道独立卷积
        grad_loss = 0.0
        channels = pred.shape[1]  # C

        for c in range(channels):
            pred_c = pred[:, c:c+1, :, :]   # [B, 1, H, W]
            target_c = target[:, c:c+1, :, :]

            pred_grad_x = F.conv2d(pred_c, self.sobel_x, padding=1, groups=1)
            pred_grad_y = F.conv2d(pred_c, self.sobel_y, padding=1, groups=1)
            target_grad_x = F.conv2d(target_c, self.sobel_x, padding=1, groups=1)
            target_grad_y = F.conv2d(target_c, self.sobel_y, padding=1, groups=1)

            grad_loss += F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

        return grad_loss / channels  # 平均所有通道的损失

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # 改成none，便于掩膜后求平均
        self.ssim = SSIMLoss()
        self.alpha = alpha

    def forward(self, pred, target, target_mask=None):
        """
        target_mask: [B,1,H,W]，1=无效（云/填充），0=有效
        """
        # 步骤1：基础有效掩膜（翻转为有效=1）
        if target_mask is not None:
            valid_mask = 1 - target_mask.float()  # 1=有效
            valid_pixels = valid_mask.sum()
            if valid_pixels == 0:
                return torch.tensor(0.0, device=pred.device)
        else:
            valid_mask = torch.ones_like(target)
            valid_pixels = valid_mask.sum()

        # 步骤2：排除精确0或0.5（填充值）
        epsilon = 1e-6
        if cfg.data.normalize_range:
            exact_fill = (torch.abs(target - 0.5) < epsilon).float()
        else:
            exact_fill = (torch.abs(target) < epsilon).float()

        # 最终掩膜：有效 且 非精确填充
        final_mask = valid_mask * (1 - exact_fill)
        final_valid_pixels = final_mask.sum()
        if final_valid_pixels == 0:
            return torch.tensor(0.0, device=pred.device)

        masked_pred = pred * final_mask
        masked_target = target * final_mask

        # MSE（掩膜后求平均）
        mse_loss = self.mse(masked_pred, masked_target).sum() / final_valid_pixels

        # SSIM（使用掩膜区域）
        ssim_loss = self.ssim(masked_pred, masked_target)

        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

class CombinedLossWithLogistic(nn.Module):
    def __init__(self, alpha=0.8, logistic_weight=0.01, num_classes=5, sigma=0.05,
                 embed_params=False, label_smoothing=0.01, gradient_weight=0.1,
                 adaptive_logistic=True):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ssim = SSIMLoss()
        self.gradient_loss = GradientLoss()
        self.alpha = alpha
        self.gradient_weight = gradient_weight
        self.logistic_weight = logistic_weight
        self.sigma = sigma
        self.embed_params = embed_params
        self.label_smoothing = label_smoothing
        self.adaptive_logistic = adaptive_logistic

        if embed_params:
            self.param_embed = nn.Embedding(num_classes + 1, 7, padding_idx=0)
            default_params = torch.tensor([
                [p["K1"], p["r1"], p["t_01"], p["K2"], p["r2"], p["t_02"], p["c"]]
                for _, p in sorted(LANDCOVER_CLASSES.items())
            ])
            self.param_embed.weight.data[1:] = default_params

        # 注意：这里不要放任何依赖 pred 的代码！

    def logistic_constraint(self, pred, target_dates, landcover, device, logistic_params=None, mask=None):
        # 这里才有 pred，可以安全解包形状
        batch_size, channels, height, width = pred.shape
        total_loss = torch.tensor(0.0, device=device)
        range_loss = torch.tensor(0.0, device=device)

        # target_dates 处理（兼容 list 或 list[list]）
        if target_dates and isinstance(target_dates[0], list):
            target_dates_flat = [d[0] for d in target_dates]
        else:
            target_dates_flat = target_dates or []

        if not target_dates_flat:
            return torch.tensor(0.0, device=device)

        target_dates_np = [np.datetime64(d, 'D').astype(datetime) for d in target_dates_flat]
        t_days = torch.tensor([float(get_doy(np.datetime64(d))) for d in target_dates_np], device=device)

        for class_id, params in LANDCOVER_CLASSES.items():
            if not params["use_logistic"] or not params["use_double_logistic"]:
                continue
            class_mask = (landcover == class_id).float()
            if class_mask.sum() == 0:
                continue

            # 参数获取（原逻辑）
            if self.adaptive_logistic and logistic_params is not None:
                params_tensor = logistic_params[:, class_id-1, :]
                K1, r1, t_01, K2, r2, t_02, c = torch.unbind(params_tensor, dim=-1)
                K1 = K1.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)
                r1 = r1.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)
                t_01 = t_01.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)
                K2 = K2.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)
                r2 = r2.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)
                t_02 = t_02.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)
                c = c.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)
            else:
                # 固定参数或嵌入（保持原样）
                if self.embed_params:
                    embed = self.param_embed(torch.tensor(class_id, device=device))
                    K1, r1, t_01, K2, r2, t_02, c = embed
                else:
                    K1 = torch.tensor(params["K1"], device=device)
                    r1 = torch.tensor(params["r1"], device=device)
                    t_01 = torch.tensor(params["t_01"], device=device)
                    K2 = torch.tensor(params["K2"], device=device)
                    r2 = torch.tensor(params["r2"], device=device)
                    t_02 = torch.tensor(params["t_02"], device=device)
                    c = torch.tensor(params["c"], device=device)
                K1 = K1.view(1, 1, 1, 1).expand(batch_size, channels, height, width)
                r1 = r1.view(1, 1, 1, 1).expand(batch_size, channels, height, width)
                t_01 = t_01.view(1, 1, 1, 1).expand(batch_size, channels, height, width)
                K2 = K2.view(1, 1, 1, 1).expand(batch_size, channels, height, width)
                r2 = r2.view(1, 1, 1, 1).expand(batch_size, channels, height, width)
                t_02 = t_02.view(1, 1, 1, 1).expand(batch_size, channels, height, width)
                c = c.view(1, 1, 1, 1).expand(batch_size, channels, height, width)

            # 逐样本计算
            for b in range(batch_size):
                t = t_days[b].view(1, 1, 1, 1).expand(1, channels, height, width)
                expected_ndvi = c[b:b+1] + K1[b:b+1] / (1 + torch.exp(-r1[b:b+1] * (t - t_01[b:b+1]))) + \
                                K2[b:b+1] / (1 + torch.exp(-r2[b:b+1] * (t - t_02[b:b+1])))

                mask_b = class_mask[b:b+1]
                if mask is not None:
                    mask_b = mask_b * mask[b:b+1]

                if mask_b.sum() == 0:
                    continue

                pred_b = pred[b:b+1]
                logistic_loss_b = torch.mean(mask_b * (pred_b - expected_ndvi) ** 2)
                total_loss += logistic_loss_b

                lower = expected_ndvi - self.sigma
                upper = expected_ndvi + self.sigma
                out_of_range = torch.clamp(pred_b - upper, min=0) + torch.clamp(lower - pred_b, min=0)
                range_loss += torch.mean(mask_b * out_of_range ** 2)

        return total_loss + range_loss


    def forward(self, pred, target, landcover, target_dates, target_mask=None, logistic_params=None):
        # 步骤1：基础有效掩膜
        if target_mask is not None:
            valid_mask = 1 - target_mask.float()  # 1=有效
            valid_pixels = valid_mask.sum()
            if valid_pixels == 0:
                return torch.tensor(0.0, device=pred.device)
        else:
            valid_mask = torch.ones_like(target)
            valid_pixels = valid_mask.sum()

        # 步骤2：排除精确0或0.5
        epsilon = 1e-6
        if cfg.data.normalize_range:
            exact_fill = (torch.abs(target - 0.5) < epsilon).float()
        else:
            exact_fill = (torch.abs(target) < epsilon).float()

        final_mask = valid_mask * (1 - exact_fill)
        final_valid_pixels = final_mask.sum()
        if final_valid_pixels == 0:
            return torch.tensor(0.0, device=pred.device)

        masked_pred = pred * final_mask
        masked_target = target * final_mask

        # MSE
        mse_loss = self.mse(masked_pred, masked_target).sum() / final_valid_pixels

        # SSIM
        ssim_loss = self.ssim(masked_pred, masked_target)

        # Gradient
        gradient = self.gradient_loss(masked_pred[:, :1, :, :], masked_target[:, :1, :, :]) 

        # 基础损失
        base_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss + self.gradient_weight * gradient

        # 标签平滑（可选）
        if self.label_smoothing > 0:
            smooth_target = masked_target * (1 - self.label_smoothing) + self.label_smoothing * 0.5
            smooth_mse = self.mse(masked_pred, smooth_target).sum() / final_valid_pixels
            base_loss = 0.5 * base_loss + 0.5 * smooth_mse

        # Logistic
        total_loss = base_loss
        if self.logistic_weight > 0:
            logistic_loss = self.logistic_constraint(
                pred, target_dates, landcover, pred.device, logistic_params, mask=final_mask
            )
            total_loss += self.logistic_weight * logistic_loss

        return total_loss

class MultiModalFusionNet(nn.Module):
    def __init__(
        self,
        modis_bands,
        landsat_bands,
        max_modis_per_target=20,
        max_landsat_per_target=3,
        use_attention=True,
        use_transformer=True,
        use_landcover=True,
        num_landcover_classes=5,
        mc_dropout=False,
        use_multiscale=True,
        adaptive_logistic=True
    ):
        super().__init__()
        self.modis_bands = modis_bands
        self.landsat_bands = landsat_bands
        self.max_modis_per_target = max_modis_per_target
        self.max_landsat_per_target = max_landsat_per_target
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        self.use_landcover = use_landcover
        self.mc_dropout = mc_dropout
        self.use_multiscale = use_multiscale
        self.adaptive_logistic = adaptive_logistic

        input_channels = (
            modis_bands * max_modis_per_target
            + landsat_bands * max_landsat_per_target
            + 1
        )

        # ---------- MODIS → Landsat ----------
        self.modis_to_landsat = nn.Sequential(
            nn.Conv2d(modis_bands, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, landsat_bands, 1)
        )

        # ---------- Initial feature ----------
        if self.use_multiscale:
            self.init_conv = MultiScaleFeatureExtractor(input_channels, 64)
        else:
            self.init_conv = nn.Sequential(
                nn.Conv2d(input_channels, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

        # ---------- Missing branch ----------
        self.initial_missing_branch = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # ---------- Time encoder ----------
        if use_transformer:
            self.time_encoder = TransformerTimeEncoder(input_dim=65, model_dim=64)
        else:
            self.time_encoder = nn.Sequential(
                nn.Linear(65, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
        self.month_embedding = nn.Embedding(13, 64)

        # ---------- Landcover ----------
        if use_landcover:
            self.lc_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        else:
            self.lc_encoder = None

        # ---------- Attention ----------
        self.att3 = AttentionBlock(128, 128, 64) if use_attention else nn.Identity()
        self.att2 = AttentionBlock(64, 64, 32) if use_attention else nn.Identity()

        TIME_CHANNELS = 64
        LC_CHANNELS = 64
        if use_landcover:
            self.lc_attn2 = TemporalAndLandcoverGuidedAttention(
                64, LC_CHANNELS, TIME_CHANNELS, 32
            )
            self.lc_attn3 = TemporalAndLandcoverGuidedAttention(
                128, LC_CHANNELS, TIME_CHANNELS, 64
            )
        else:
            self.lc_attn2 = nn.Identity()
            self.lc_attn3 = nn.Identity()

        enc_in = 64 + (LC_CHANNELS if use_landcover else 0)

        # ---------- Encoder ----------
        self.enc1 = self._make_encoder(enc_in, 64)   # 1/2
        self.enc2 = self._make_encoder(64, 128)     # 1/4
        self.enc3 = self._make_encoder(128, 256)    # 1/8

        # ---------- Decoder ----------
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._make_decoder(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._make_decoder(128, 64)
        # >>> MODIFIED: 新增 dec1，直接恢复到原分辨率
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = self._make_decoder(128, 64)

        # ---------- Dropout ----------
        self.mc_dropout1 = nn.Dropout2d(0.2) if mc_dropout else nn.Identity()
        self.mc_dropout2 = nn.Dropout2d(0.2) if mc_dropout else nn.Identity()
        self.mc_dropout3 = nn.Dropout2d(0.2) if mc_dropout else nn.Identity()

        # ---------- Final ----------
        final_in = 64 + 32 + landsat_bands
        self.final = nn.Sequential(
            nn.Conv2d(final_in, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, landsat_bands, 1),
            nn.Sigmoid()
        )
        self.missing_pred_head = nn.Conv2d(32, 1, 1)

        # 修改这里：logistic_predictor 的输入通道固定使用 init_conv 的输出通道（64），而不是 enc_in
        if adaptive_logistic:
            # 关键改动：用 64 而不是 enc_in
            self.logistic_predictor = AdaptiveLogisticPredictor(64)
    def _make_encoder(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def _make_decoder(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x, lc, dates):
        B, C, H, W = x.shape
        modis_feat = x[:, :self.modis_bands * self.max_modis_per_target]
        landsat_feat = x[:, self.modis_bands * self.max_modis_per_target:-1]
        missing_mask = x[:, -1:]

        # ----- Time embedding -----
        modis_diffs = dates['modis_diffs'].to(x.device)
        landsat_diffs = dates['landsat_diffs'].to(x.device)
        modis_months = dates['modis_months'].long().to(x.device)
        landsat_months = dates['landsat_months'].long().to(x.device)

        modis_time = torch.cat(
            [modis_diffs.unsqueeze(-1), self.month_embedding(modis_months)], dim=-1
        )
        landsat_time = torch.cat(
            [landsat_diffs.unsqueeze(-1), self.month_embedding(landsat_months)], dim=-1
        )

        modis_time = self.time_encoder(modis_time)
        landsat_time = self.time_encoder(landsat_time)
        time_embed = ((modis_time + landsat_time) / 2).view(B, -1, 1, 1)

        # 准备广播用的 time_embed（供 lc_attn 使用）
        time_embed_broadcasted = time_embed.expand(-1, -1, H, W)

        # ----- Init -----
        x_feat = self.init_conv(torch.cat([modis_feat, landsat_feat, missing_mask], 1))

        lc_feat = None
        if self.use_landcover:
            lc_feat = self.lc_encoder(lc)

        if self.adaptive_logistic:
            logistic_params = self.logistic_predictor(x_feat)
        else:
            logistic_params = None

        # ----- Encoder -----
        e1 = self.enc1(x_feat if lc_feat is None else torch.cat([x_feat, lc_feat], 1))
        e1 = self.mc_dropout1(e1)

        e2 = self.enc2(e1)
        e2 = self.mc_dropout2(e2)

        e3 = self.enc3(e2)

        # ----- Decoder -----
        d3 = self.dec3(torch.cat([self.up3(e3), e2], 1))

        # 加入 landcover + time 引导注意力（如果启用）
        if self.use_landcover and lc_feat is not None:
            d3 = self.lc_attn3(d3, lc_feat, time_embed_broadcasted)

        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))

        if self.use_landcover and lc_feat is not None:
            d2 = self.lc_attn2(d2, lc_feat, time_embed_broadcasted)

        # >>> MODIFIED: 直接恢复到 d1
        d1 = self.dec1(torch.cat([self.up1(d2), x_feat[:, :64]], 1))

        missing_feat = self.initial_missing_branch(missing_mask)

        modis_filled = torch.mean(
            torch.stack([
                self.modis_to_landsat(
                    modis_feat[:, i*self.modis_bands:(i+1)*self.modis_bands]
                )
                for i in range(self.max_modis_per_target)
            ], dim=1),
            dim=1
        )

        nearest_landsat = landsat_feat[:, -self.landsat_bands:]
        filled_feat = nearest_landsat * (1 - missing_mask) + modis_filled * missing_mask

        out = self.final(torch.cat([d1, missing_feat, filled_feat], 1))
        missing_pred = self.missing_pred_head(missing_feat)

        return out.unsqueeze(1), missing_pred, logistic_params
class PairedFusionDataset(Dataset):
    def __init__(self, blocks, region_ids=None, split='train'):
        self.X = blocks['X']
        self.Y = blocks['Y']
        self.landcover_blocks = blocks['landcover'] if 'landcover' in blocks else None
        self.positions = blocks['positions']
        self.target_masks = blocks['target_masks']
        self.modis_diffs = blocks['modis_diffs']
        self.landsat_diffs = blocks['landsat_diffs']
        self.modis_months = blocks['modis_months']
        self.landsat_months = blocks['landsat_months']
        self.target_dates = blocks['target_dates']
        self.region_ids = region_ids if region_ids is not None else [0] * len(self.X)  # 新增：区域ID
        self.full_height = blocks['full_height']
        self.full_width = blocks['full_width']
        self.split = split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            "X": self.X[idx],
            "Y": self.Y[idx],
            "positions": self.positions[idx],
            "landcover": self.landcover_blocks[idx] if self.landcover_blocks else None,
            "target_masks": self.target_masks[idx],
            "modis_diffs": self.modis_diffs[idx],
            "landsat_diffs": self.landsat_diffs[idx],
            "modis_months": self.modis_months[idx],
            "landsat_months": self.landsat_months[idx],
            "target_dates": [self.target_dates[idx]],
            "region_id": self.region_ids[idx]  # 新增
        }
        return sample

def create_fusion_blocks(modis_paths, landsat_paths, landcover_path=None, block_size=64, overlap=16, augment=False, target_paths=None):
    X = []
    Y = []
    landcover_blocks = []
    positions = []
    target_masks = []
    modis_diffs = []
    landsat_diffs = []
    modis_months = []
    landsat_months = []
    target_dates_list = []
    region_ids = []

    block_size = cfg.data.block_size
    stride = block_size - overlap

    modis_paths = sorted(modis_paths, key=parse_date_from_filename)
    landsat_paths = sorted(landsat_paths, key=parse_date_from_filename)

    modis_dates = [parse_date_from_filename(p) for p in modis_paths]
    landsat_dates = [parse_date_from_filename(p) for p in landsat_paths]

    modis_list_full = [load_raster(p, ref_path=landsat_paths[0])[0] for p in modis_paths]
    modis_masks_full = [load_raster(p, ref_path=landsat_paths[0])[2][0] for p in modis_paths]
    landsat_list_full = [load_raster(p)[0] for p in landsat_paths]
    landsat_masks_full = [load_raster(p)[2][0] for p in landsat_paths]

    print(f"MODIS bands: {modis_list_full[0].shape[0]}")
    print(f"Landsat bands: {landsat_list_full[0].shape[0]}")
    logging.info(f"MODIS bands: {modis_list_full[0].shape[0]}, Landsat bands: {landsat_list_full[0].shape[0]}")

    landcover_data = None
    if landcover_path and os.path.exists(landcover_path):
        landcover_data = load_raster(landcover_path, ref_path=landsat_paths[0])[0]
    else:
        print(f"警告: landcover 文件不可用，填充零值 landcover_blocks")
        logging.warning("landcover 文件不可用，填充零值 landcover_blocks")

    _, h, w = landsat_list_full[0].shape

    positions_template = []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            actual_h = min(block_size, h - i)
            actual_w = min(block_size, w - j)
            positions_template.append((i, j, actual_h, actual_w))

    if target_paths is not None:
        target_paths = sorted(target_paths, key=parse_date_from_filename)
    else:
        target_paths = landsat_paths

    target_dates = [parse_date_from_filename(p) for p in target_paths]
    target_indices = [landsat_paths.index(p) for p in target_paths if p in landsat_paths]

    num_regions = 2  # 假设4个区域
    num_rows = int(np.sqrt(num_regions))
    num_cols = num_regions // num_rows

    for idx_target, target_date in enumerate(target_dates):
        target_path = target_paths[idx_target]

        filtered_modis_paths, filtered_modis_dates = filter_by_window(
            modis_paths, modis_dates, target_date, cfg.data.window_days, None, exclude_target=False
        )

        filtered_landsat_paths, filtered_landsat_dates = filter_by_window(
            landsat_paths, landsat_dates, target_date, cfg.data.window_days, cfg.data.max_landsat_per_target, exclude_target=True
        )

        if len(filtered_landsat_paths) == 0:
            print(f"跳过目标日期 {target_date}: 窗口期内没有 Landsat 影像")
            logging.info(f"跳过目标日期 {target_date}: 窗口期内没有 Landsat 影像")
            continue

        # ========== 新逻辑：MODIS 选择 ==========
        # 步骤1：必选 - 为每个 Landsat 日期（包括目标日期）找最近的 MODIS
        anchor_dates = filtered_landsat_dates + [target_date]  # 包括目标日期
        closest_modis = []
        for ls_date in anchor_dates:
            if filtered_modis_dates:  # 防止空列表
                closest_date = min(filtered_modis_dates, key=lambda md: abs(calculate_day_diff(md, ls_date)))
                idx = filtered_modis_dates.index(closest_date)
                closest_modis.append((filtered_modis_paths[idx], closest_date))

        # 去重（同一日期可能被多个 anchor 选中）
        closest_modis = list(dict.fromkeys(closest_modis))  # 保持顺序去重

        num_closest = len(closest_modis)

        # 步骤2：如果不足 max，在窗口内均匀补充剩余
        remaining = cfg.data.max_modis_per_target - num_closest
        selected_even = []
        if remaining > 0 and filtered_modis_dates:
            # 排序时间
            sorted_dates = sorted(filtered_modis_dates)
            sorted_paths = [filtered_modis_paths[filtered_modis_dates.index(d)] for d in sorted_dates]

            # 排除已选的 closest
            available_dates = [d for d in sorted_dates if d not in [d for _, d in closest_modis]]
            available_paths = [p for p, d in zip(sorted_paths, sorted_dates) if d in available_dates]

            if len(available_dates) >= remaining:
                # 均匀选点
                indices = np.linspace(0, len(available_dates) - 1, remaining, dtype=int)
                selected_even = [(available_paths[i], available_dates[i]) for i in indices]
            else:
                # 如果不够，全部补上
                selected_even = list(zip(available_paths, available_dates))

        # 最终 MODIS
        final_modis = closest_modis + selected_even
        filtered_modis_paths = [p for p, _ in final_modis]
        filtered_modis_dates = [d for _, d in final_modis]

        # ========== 打印 ==========
        print(f"目标日期 {target_date}: 必选最近MODIS日期: {[d for _, d in closest_modis]}")
        print(f"目标日期 {target_date}: 均匀补充MODIS日期: {[d for _, d in selected_even]}")
        print(f"目标日期 {target_date}: 最终MODIS {len(filtered_modis_paths)} 张, Landsat {len(filtered_landsat_paths)} 张")

        logging.info(f"目标日期 {target_date}: 必选最近MODIS日期: {[d for _, d in closest_modis]}")
        logging.info(f"目标日期 {target_date}: 均匀补充MODIS日期: {[d for _, d in selected_even]}")
        logging.info(f"目标日期 {target_date}: 最终MODIS {len(filtered_modis_paths)} 张, Landsat {len(filtered_landsat_paths)} 张")

        # ========== 后续代码不变 ==========
        modis_indices = [modis_paths.index(p) for p in filtered_modis_paths]
        landsat_indices = [landsat_paths.index(p) for p in filtered_landsat_paths]

        modis_diff = [calculate_day_diff(d, target_date) for d in filtered_modis_dates]
        landsat_diff = [calculate_day_diff(d, target_date) for d in filtered_landsat_dates]

        modis_diff_t = torch.tensor(modis_diff, dtype=torch.float32)
        landsat_diff_t = torch.tensor(landsat_diff, dtype=torch.float32)
        modis_months_t = torch.tensor([get_month(d) for d in filtered_modis_dates], dtype=torch.long)
        landsat_months_t = torch.tensor([get_month(d) for d in filtered_landsat_dates], dtype=torch.long)
        label_data_single, _, label_mask_single = load_raster(target_path)
        label_data_single = [label_data_single]
        label_masks_single = [label_mask_single[0]]
        valid_blocks = 0
        invalid_blocks = 0
        for pos_idx, pos in enumerate(positions_template):
            i, j, actual_h, actual_w = pos
            # 计算区域ID（空间分层）
            region_row = min(int(i / h * num_rows), num_rows - 1)
            region_col = min(int(j / w * num_cols), num_cols - 1)
            region_id = region_row * num_cols + region_col
            try:
                modis_blocks = [modis_list_full[k][:, i:i+actual_h, j:j+actual_w] for k in modis_indices]
                landsat_blocks = [landsat_list_full[k][:, i:i+actual_h, j:j+actual_w] for k in landsat_indices]
                label_blocks = [l[:, i:i+actual_h, j:j+actual_w] for l in label_data_single]
                modis_block_masks = [modis_masks_full[k][i:i+actual_h, j:j+actual_w] for k in modis_indices]
                landsat_block_masks = [landsat_masks_full[k][i:i+actual_h, j:j+actual_w] for k in landsat_indices]
                label_block_masks = [m[i:i+actual_h, j:j+actual_w] for m in label_masks_single]
                valid_mask = np.ones((actual_h, actual_w), dtype=np.uint8)
                for mask in modis_block_masks + landsat_block_masks:
                    valid_mask &= mask
                if augment and np.random.rand() < (0.5 if get_month(target_date) in [6, 7, 8] else 0.3):
                    cloud_ratio = np.random.uniform(0.2, 0.4 if get_month(target_date) in [6, 7, 8] else 0.3)
                    cloud_mask = np.random.rand(actual_h, actual_w) < cloud_ratio
                    for block in modis_blocks + landsat_blocks:
                        block[:, cloud_mask] = 0
                    valid_mask[cloud_mask] = 0
                    label_block_masks = [m | cloud_mask for m in label_block_masks]
                if np.sum(valid_mask) == 0:
                    print(f"目标日期 {target_date} 块 {pos} 无效: 所有像素被掩膜")
                    logging.info(f"目标日期 {target_date} 块 {pos} 无效: 所有像素被掩膜")
                    invalid_blocks += 1
                    continue
                input_blocks = np.concatenate([
                    np.stack(modis_blocks).reshape(-1, actual_h, actual_w),
                    np.stack(landsat_blocks).reshape(-1, actual_h, actual_w)
                ], axis=0)
                missing_mask = 1 - valid_mask
                input_blocks = np.concatenate([input_blocks, missing_mask[np.newaxis, :, :]], axis=0)
                label_blocks = [np.where(m, l, 0) for m, l in zip(label_block_masks, label_blocks)]
                target_mask = [1 - m for m in label_block_masks]
                if actual_h < block_size or actual_w < block_size:
                    pad_h = block_size - actual_h
                    pad_w = block_size - actual_w
                    data_channels = input_blocks.shape[0] - 1
                    data_part = input_blocks[:data_channels]
                    mask_part = input_blocks[data_channels:]
                    data_padded = np.pad(data_part, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                    mask_padded = np.pad(mask_part, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=1)
                    input_blocks = np.concatenate([data_padded, mask_padded], axis=0)
                    label_blocks = [np.pad(l, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0) for l in label_blocks]
                    target_mask = [np.pad(m, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1) for m in target_mask]
                # 验证块形状
                if input_blocks.shape[1:] != (block_size, block_size):
                    logging.warning(f"目标日期 {target_date} 块 {pos}: input_blocks 形状 {input_blocks.shape[1:]}, 预期 ({block_size}, {block_size})")
                if label_blocks[0].shape[1:] != (block_size, block_size):
                    logging.warning(f"目标日期 {target_date} 块 {pos}: label_blocks 形状 {label_blocks[0].shape[1:]}, 预期 ({block_size}, {block_size})")
                X.append(torch.tensor(input_blocks, dtype=torch.float32))
                Y.append(torch.tensor(np.stack(label_blocks), dtype=torch.float32))
                target_masks.append(torch.tensor(np.stack(target_mask), dtype=torch.float32))
                positions.append(pos)
                if landcover_data is not None:
                    lc_block = landcover_data[:, i:i+actual_h, j:j+actual_w]
                    if actual_h < block_size or actual_w < block_size:
                        lc_block = np.pad(lc_block, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                    landcover_blocks.append(torch.tensor(lc_block, dtype=torch.float32))
                else:
                    landcover_blocks.append(torch.zeros(1, block_size, block_size, dtype=torch.float32))
                modis_diffs.append(modis_diff_t)
                landsat_diffs.append(landsat_diff_t)
                modis_months.append(modis_months_t)
                landsat_months.append(landsat_months_t)
                target_dates_list.append(str(target_date))
                region_ids.append(region_id)  # 新增
                valid_blocks += 1
            except Exception as e:
                print(f"处理目标日期 {target_date} 块 {pos} 时出错: {str(e)}")
                logging.error(f"处理目标日期 {target_date} 块 {pos} 时出错: {str(e)}")
                invalid_blocks += 1
                continue
        print(f"目标日期 {target_date}: 有效块数 {valid_blocks}, 无效块数 {invalid_blocks}")
        logging.info(f"目标日期 {target_date}: 有效块数 {valid_blocks}, 无效块数 {invalid_blocks}")
    print(f"X blocks: {len(X)}, Y blocks: {len(Y)}, positions: {len(positions)}, "
          f"target_masks: {len(target_masks)}, landcover_blocks: {len(landcover_blocks)}")
    logging.info(f"X blocks: {len(X)}, Y blocks: {len(Y)}, positions: {len(positions)}, "
                 f"target_masks: {len(target_masks)}, landcover_blocks: {len(landcover_blocks)}")
    return {
        'X': X,
        'Y': Y,
        'landcover': landcover_blocks,
        'positions': positions,
        'target_masks': target_masks,
        'modis_diffs': modis_diffs,
        'landsat_diffs': landsat_diffs,
        'modis_months': modis_months,
        'landsat_months': landsat_months,
        'target_dates': target_dates_list,
        'region_ids': region_ids,
        'full_height': h,
        'full_width': w
    }

def fusion_collate_fn(batch):
    xs = torch.stack([item["X"] for item in batch])
    ys = torch.stack([item["Y"] for item in batch])
    lcs = torch.stack([item["landcover"] for item in batch])
    positions = [item["positions"] for item in batch]
    target_masks = torch.stack([item["target_masks"] for item in batch])
    dates = {
        "modis_diffs": torch.stack([item["modis_diffs"] for item in batch]),
        "landsat_diffs": torch.stack([item["landsat_diffs"] for item in batch]),
        "modis_months": torch.stack([item["modis_months"] for item in batch]),
        "landsat_months": torch.stack([item["landsat_months"] for item in batch])
    }
    target_dates = [item["target_dates"][0] for item in batch]
    region_ids = [item["region_id"] for item in batch]  # 新增
    return xs, ys, lcs, positions, dates, target_masks, target_dates, region_ids

def train_epoch(model, loader, device, optimizer, criterion, gradient_accum_steps=1, epoch=0, total_epochs=1, logistic_scheduler=None, scaler=None):
    model.train()
    total_loss = 0
    accum_count = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        x, y, lc, positions, dates, target_masks, target_dates, _ = batch
        x, y, lc, target_masks = x.to(device), y.to(device), lc.to(device), target_masks.to(device)
        dates = {k: v.to(device).float() if k in ['modis_diffs', 'landsat_diffs'] else v.to(device).long() for k, v in dates.items()}
        if cfg.training.use_amp and scaler:
            with torch.amp.autocast('cuda'):
                preds, missing_preds, logistic_params = model(x, lc, dates)
                if isinstance(criterion, CombinedLossWithLogistic):
                    criterion.logistic_weight = logistic_scheduler(epoch, total_epochs)
                    main_loss = criterion(preds[:, 0], y[:, 0], lc, target_dates, target_masks[:, 0], logistic_params)
                else:
                    main_loss = criterion(preds[:, 0], y[:, 0])
                missing_loss = F.binary_cross_entropy_with_logits(missing_preds.squeeze(1), target_masks[:, 0].float())
                loss = main_loss + 0.2 * missing_loss
        else:
            preds, missing_preds, logistic_params = model(x, lc, dates)
            if isinstance(criterion, CombinedLossWithLogistic):
                criterion.logistic_weight = logistic_scheduler(epoch, total_epochs)
                main_loss = criterion(preds[:, 0], y[:, 0], lc, target_dates, target_masks[:, 0], logistic_params)
            else:
                main_loss = criterion(preds[:, 0], y[:, 0])
            missing_loss = F.binary_cross_entropy_with_logits(missing_preds.squeeze(1), target_masks[:, 0].float())
            loss = main_loss + 0.2 * missing_loss
        loss = loss / gradient_accum_steps
        if cfg.training.use_amp and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_count += 1
        if accum_count == gradient_accum_steps or (batch_idx + 1 == len(loader)):
            if cfg.training.use_amp and scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
        total_loss += loss.item() * gradient_accum_steps
    return total_loss / len(loader)

def evaluate(model, loader, device, criterion, output_dir=None, experiment_name=""):
    model.eval()
    total_ssim = 0
    total_psnr = 0
    total_mae = 0
    total_samples = 0
    date_preds = defaultdict(list)
    date_trues = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x, y, lc, positions, dates, target_masks, target_dates, _ = batch
            x, y, lc, target_masks = x.to(device), y.to(device), lc.to(device), target_masks.to(device)
            dates = {k: v.to(device).float() if k in ['modis_diffs', 'landsat_diffs'] else v.to(device).long() for k, v in dates.items()}

            preds_norm, _, _ = model(x, lc, dates)
            preds = denormalize_ndvi(preds_norm[:, 0].cpu().numpy())
            y_denorm = denormalize_ndvi(y[:, 0].cpu().numpy())

            # 基础有效掩膜
            valid_mask = (1 - target_masks[:, 0].cpu().numpy()).astype(bool)  # 1=有效

            # 排除精确0或0.5
            epsilon = 1e-6
            exact_fill = np.abs(y_denorm) < epsilon
            valid_mask = valid_mask & ~exact_fill

            for b in range(len(target_dates)):
                date = target_dates[b]
                for pred_band, gt_band, mask_band in zip(preds[b], y_denorm[b], valid_mask[b]):
                    valid_pixels = mask_band
                    pred_valid = pred_band[valid_pixels].flatten()
                    gt_valid = gt_band[valid_pixels].flatten()
                    date_preds[date].extend(pred_valid)
                    date_trues[date].extend(gt_valid)

            for pred, gt, mask in zip(preds, y_denorm, valid_mask):
                valid_pixels = mask.squeeze()
                if len(valid_pixels.shape) > 2:
                    valid_pixels = np.all(valid_pixels, axis=0).astype(bool)
                pred_valid = pred[:, valid_pixels]
                gt_valid = gt[:, valid_pixels]


                # 在 evaluate 函数的 ssim 计算循环中修改为：
                ssim_scores = []
                for p_band, g_band in zip(pred_valid, gt_valid):
                    if len(p_band) < 7 or len(g_band) < 7:
                        continue  # 跳过像素太少的区域，避免报错
    
                    try:
                        ssim_val = ssim(p_band, g_band, data_range=2.0, win_size=5)  # 显式设小窗口 5×5
                        ssim_scores.append(ssim_val)
                    except ValueError as e:
                        print(f"SSIM 警告: {e}，跳过该区域")
                        continue

                ssim_val = np.mean(ssim_scores) if ssim_scores else 0

                psnr_scores = [psnr(p_band, g_band, data_range=2.0) for p_band, g_band in zip(pred_valid, gt_valid)]
                psnr_val = np.mean(psnr_scores) if psnr_scores else 0
                mae = np.mean(np.abs(pred_valid - gt_valid))

                total_ssim += ssim_val
                total_psnr += psnr_val
                total_mae += mae
                total_samples += 1
    date_r2s = []
    date_rmses = []
    for date in date_preds:
        preds_list = np.array(date_preds[date])
        trues_list = np.array(date_trues[date])
        if len(preds_list) > 0:
            date_rmse = np.sqrt(mean_squared_error(trues_list, preds_list))
            date_r2 = r2_score(trues_list, preds_list) if np.var(trues_list) > 0 else 0
            date_r2s.append(date_r2)
            date_rmses.append(date_rmse)
            print(f"Date {date}: RMSE = {date_rmse:.4f}, R2 = {date_r2:.4f}")
            logging.info(f"Date {date}: RMSE = {date_rmse:.4f}, R2 = {date_r2:.4f}")
    avg_rmse = np.mean(date_rmses) if date_rmses else float('inf')
    avg_r2 = np.mean(date_r2s) if date_r2s else 0
    avg_ssim = total_ssim / total_samples if total_samples > 0 else 0
    avg_psnr = total_psnr / total_samples if total_samples > 0 else 0
    avg_mae = total_mae / total_samples if total_samples > 0 else 0
    all_preds = []
    all_trues = []
    for date in date_preds:
        all_preds.extend(date_preds[date])
        all_trues.extend(date_trues[date])
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    return avg_rmse, avg_r2, avg_ssim, avg_psnr, avg_mae, all_preds, all_trues

def evaluate_per_date(model, loader, device):
    model.eval()
    date_preds = defaultdict(list)
    date_trues = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            x, y, lc, _, dates, target_masks, target_dates, _ = batch
            x, y, lc, target_masks = x.to(device), y.to(device), lc.to(device), target_masks.to(device)
            dates = {k: v.to(device).float() if k in ['modis_diffs', 'landsat_diffs'] else v.to(device).long() for k, v in dates.items()}

            preds_norm, _, _ = model(x, lc, dates)
            preds = denormalize_ndvi(preds_norm[:, 0].cpu().numpy())
            y_denorm = denormalize_ndvi(y[:, 0].cpu().numpy())

            # 基础有效掩膜
            valid_mask = (1 - target_masks[:, 0].cpu().numpy()).astype(bool)

            # 排除精确0或0.5
            epsilon = 1e-6
            exact_fill = np.abs(y_denorm) < epsilon
            valid_mask = valid_mask & ~exact_fill

            for b in range(len(target_dates)):
                date = target_dates[b]
                for pred_band, gt_band, mask_band in zip(preds[b], y_denorm[b], valid_mask[b]):
                    valid_pixels = mask_band
                    pred_valid = pred_band[valid_pixels].flatten()
                    gt_valid = gt_band[valid_pixels].flatten()
                    date_preds[date].extend(pred_valid)
                    date_trues[date].extend(gt_valid)

    date_rmse = {}
    date_r2 = {}
    for date in date_preds:
        preds_list = np.array(date_preds[date])
        trues_list = np.array(date_trues[date])
        if len(preds_list) > 0:
            date_rmse[date] = np.sqrt(mean_squared_error(trues_list, preds_list))
            date_r2[date] = r2_score(trues_list, preds_list) if np.var(trues_list) > 0 else 0

    return date_rmse, date_r2
def logistic_weight_scheduler(epoch, total_epochs, start=0.01, end=0.1):
    return start + (end - start) * (epoch / total_epochs)

def generate_weight_matrix(size, overlap):
    weight = np.ones((size, size), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, endpoint=False)
        weight[:, :overlap] *= ramp[np.newaxis, :]
        weight[:, -overlap:] *= ramp[np.newaxis, ::-1]
        weight[:overlap, :] *= ramp[:, np.newaxis]
        weight[-overlap:, :] *= ramp[::-1, np.newaxis]
    return weight

def mc_dropout_predict(model, loader, device, num_samples=10):
    model.eval()
    model.apply(lambda m: m.train() if isinstance(m, nn.Dropout2d) else None)
    all_predictions = []
    all_positions = []
    all_trues = []
    block_size = cfg.data.block_size
    with torch.no_grad():
        for batch in tqdm(loader, desc="MC Dropout Prediction"):
            x, y, lc, positions, dates, target_masks, target_dates, _ = batch
            x, lc, y = x.to(device), lc.to(device), y.to(device)
            dates = {k: v.to(device).float() if k in ['modis_diffs', 'landsat_diffs'] else v.to(device).long() for k, v in dates.items()}
            batch_predictions = []
            for _ in range(num_samples):
                preds_norm, _, _ = model(x, lc, dates)
                preds = denormalize_ndvi(preds_norm[:, 0].cpu().numpy())
                batch_predictions.append(preds)
            batch_predictions = np.stack(batch_predictions)
            all_predictions.append(batch_predictions)
            all_positions.extend(positions)
            true_vals = denormalize_ndvi(y[:, 0].cpu().numpy())
            all_trues.append(true_vals)
    all_predictions = np.concatenate(all_predictions, axis=1)
    all_trues = np.concatenate(all_trues, axis=0)
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)
    return mean_pred, std_pred, all_trues, all_positions

def reconstruct_full_image(predictions, positions, full_height, full_width, block_size=64, overlap=16):
    block_size = cfg.data.block_size
    num_bands = predictions.shape[1]
    full_image = np.zeros((num_bands, full_height, full_width), dtype=np.float32)
    weight_sum = np.zeros((num_bands, full_height, full_width), dtype=np.float32)
    weight = generate_weight_matrix(block_size, overlap)
    for pred, pos in zip(predictions, positions):
        i, j, actual_h, actual_w = pos
        if pred.shape[1:] != (actual_h, actual_w):
            logging.warning(f"Block at {pos}: pred shape {pred.shape[1:]}, expected ({actual_h}, {actual_w})")
            pred = pred[:, :actual_h, :actual_w]
        for b in range(num_bands):
            block = pred[b, :actual_h, :actual_w]
            weight_block = weight[:actual_h, :actual_w]
            full_image[b, i:i+actual_h, j:j+actual_w] += block * weight_block
            weight_sum[b, i:i+actual_h, j:j+actual_w] += weight_block
    valid = weight_sum > 0
    full_image[valid] /= weight_sum[valid]
    return full_image

def predict_target_date(model, dataset, device, output_dir, experiment_name, ref_profile):
    model.eval()
    full_height = dataset.full_height
    full_width = dataset.full_width
    num_bands = dataset.Y[0].shape[0]
    sum_pred = np.zeros((num_bands, full_height, full_width), dtype=np.float32)
    sum_weight = np.zeros((full_height, full_width), dtype=np.float32)
    full_true = np.full((num_bands, full_height, full_width), np.nan)
    block_size = cfg.data.block_size
    overlap = cfg.data.overlap
    weight_matrix = generate_weight_matrix(block_size, overlap)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=fusion_collate_fn)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting target date"):
            x, y, lc, positions, dates, target_masks, target_date, _ = batch
            x, lc = x.to(device), lc.to(device)
            dates = {
                'modis_diffs': dates['modis_diffs'].to(device).float(),
                'landsat_diffs': dates['landsat_diffs'].to(device).float(),
                'modis_months': dates['modis_months'].to(device).long(),
                'landsat_months': dates['landsat_months'].to(device).long()
            }
            preds_norm, _, _ = model(x, lc, dates)
            pred = denormalize_ndvi(preds_norm[:, 0].cpu().numpy())
            y_denorm = denormalize_ndvi(y[:, 0].cpu().numpy())
            for i in range(pred.shape[0]):
                row, col, actual_h, actual_w = positions[i]
                pred_block = pred[i, :, :actual_h, :actual_w]
                true_block = y_denorm[i, :, :actual_h, :actual_w]
                weight_block = weight_matrix[:actual_h, :actual_w]
                sum_pred[:, row:row+actual_h, col:col+actual_w] += pred_block * weight_block[np.newaxis, :, :]
                sum_weight[row:row+actual_h, col:col+actual_w] += weight_block
                full_true[:, row:row+actual_h, col:col+actual_w] = true_block
    full_pred = np.where(sum_weight > 0, sum_pred / sum_weight[np.newaxis, :, :], 0)
    if ref_profile['count'] != num_bands:
        print(f"Warning: ref_profile['count'] ({ref_profile['count']}) does not match num_bands ({num_bands}). Updating profile.")
        ref_profile['count'] = num_bands
    return full_pred, full_true

def analyze_spatial_uncertainty(full_mean, full_std, full_true, output_dir, experiment_name, ref_profile=None):
    abs_error = np.abs(full_true - full_mean)
    flat_error = abs_error.flatten()
    flat_std = full_std.flatten()
    valid_mask = ~np.isnan(flat_error) & ~np.isnan(flat_std)
    if np.sum(~valid_mask) > 0:
        print(f"警告: 发现 {np.sum(~valid_mask)} 个无效像素（NaN），可能导致缺失块")
    flat_error = flat_error[valid_mask]
    flat_std = flat_std[valid_mask]
    corr_coef, p_value = pearsonr(flat_error, flat_std)
    print(f"Error-uncertainty Pearson correlation: {corr_coef:.4f} (p={p_value:.4e})")
    # 保存相关性统计
    with open(os.path.join(output_dir, f"{experiment_name}_correlation.txt"), "w") as f:
        f.write(f"Pearson Correlation Coefficient: {corr_coef}\n")
        f.write(f"P-value: {p_value}\n")
    # 定义保存函数，同时生成PNG和TIF
    def save_plot_and_tif(data, title, filename, cmap, vmin=None, vmax=None, colorbar_label=None):
        plt.figure(figsize=(12, 10))
        plt.imshow(data[0], cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar_label:
            plt.colorbar(label=colorbar_label)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        # 保存PNG
        png_path = os.path.join(output_dir, f"{experiment_name}_{filename}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved PNG: {png_path}")
        # 保存TIF
        if ref_profile is not None:
            tif_path = os.path.join(output_dir, f"{experiment_name}_{filename}.tif")
            profile = ref_profile.copy()
            if profile['count'] != 1:
                print(f"Warning: Adjusting profile count from {profile['count']} to 1 for single-band TIF")
                profile['count'] = 1
            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(data[0], 1)
            print(f"✅ Saved TIF: {tif_path}")
        else:
            print("警告: 未提供ref_profile，无法保存TIF格式")
    # 误差-不确定性散点图（仅PNG）
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=flat_std, y=flat_error, alpha=0.1, s=5)
    plt.title(f"Error vs. Uncertainty (r={corr_coef:.3f})")
    plt.xlabel("Uncertainty (Standard Deviation)")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{experiment_name}_error_vs_uncertainty.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"✅ Saved PNG: {scatter_path}")
    # 误差-不确定性密度图（仅PNG）
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=flat_std, y=flat_error, cmap="viridis", fill=True)
    plt.title(f"Error-Uncertainty Density (r={corr_coef:.3f})")
    plt.xlabel("Uncertainty (Standard Deviation)")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    density_path = os.path.join(output_dir, f"{experiment_name}_error_uncertainty_density.png")
    plt.savefig(density_path, dpi=300)
    plt.close()
    print(f"✅ Saved PNG: {density_path}")
    # 不确定性图
    save_plot_and_tif(
        full_std,
        "Spatial Uncertainty Map",
        "uncertainty_map",
        cmap='hot',
        vmin=0,
        vmax=np.percentile(full_std[0], 95),
        colorbar_label='Uncertainty (Standard Deviation)'
    )
    # 预测均值图
    save_plot_and_tif(
        full_mean,
        "Prediction Mean",
        "prediction_mean",
        cmap='viridis',
        vmin=-1,
        vmax=1,
        colorbar_label='Predicted Value'
    )
    # 真值图
    save_plot_and_tif(
        full_true,
        "True Value",
        "true_value",
        cmap='viridis',
        vmin=-1,
        vmax=1,
        colorbar_label='True Value'
    )
    # 绝对误差图
    save_plot_and_tif(
        abs_error,
        "Absolute Error Map",
        "error_map",
        cmap='Reds',
        vmin=0,
        vmax=np.percentile(abs_error[0], 95),
        colorbar_label='Absolute Error'
    )
    # 下置信区间图
    ci_lower = full_mean - 2 * full_std
    save_plot_and_tif(
        ci_lower,
        "Lower Confidence Interval Map",
        "ci_lower_map",
        cmap='cool',
        vmin=-1,
        vmax=1,
        colorbar_label='Lower Confidence Interval (mean - 2std)'
    )
    # 上置信区间图
    ci_upper = full_mean + 2 * full_std
    save_plot_and_tif(
        ci_upper,
        "Upper Confidence Interval Map",
        "ci_upper_map",
        cmap='cool',
        vmin=-1,
        vmax=1,
        colorbar_label='Upper Confidence Interval (mean + 2std)'
    )
    return corr_coef, p_value

def plot_metrics_curve(df_log, experiment_name, output_dir):
    best_idx = df_log['val_rmse'].idxmin()
    best_epoch = df_log.loc[best_idx, 'epoch']
    best_rmse = df_log.loc[best_idx, 'val_rmse']
    best_r2 = df_log.loc[best_idx, 'val_r2']
    best_ssim = df_log.loc[best_idx, 'val_ssim']
    best_psnr = df_log.loc[best_idx, 'val_psnr']
    best_mae = df_log.loc[best_idx, 'val_mae']
    best_loss = df_log.loc[best_idx, 'train_loss']
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 3, 1)
    plt.plot(df_log['epoch'], df_log['train_loss'], marker='o', color='blue', label='Train Loss')
    plt.scatter(best_epoch, best_loss, color='red', marker='*', s=150, label='Best RMSE Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.plot(df_log['epoch'], df_log['val_rmse'], marker='o', color='red', label='RMSE')
    plt.scatter(best_epoch, best_rmse, color='black', marker='*', s=150, label='Best RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.plot(df_log['epoch'], df_log['val_r2'], marker='o', color='orange', label='R²')
    plt.scatter(best_epoch, best_r2, color='black', marker='*', s=150, label='Best RMSE Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('Validation R2')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.plot(df_log['epoch'], df_log['val_ssim'], marker='o', color='green', label='SSIM')
    plt.scatter(best_epoch, best_ssim, color='black', marker='*', s=150, label='Best RMSE Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 5)
    plt.plot(df_log['epoch'], df_log['val_psnr'], marker='o', color='purple', label='PSNR')
    plt.scatter(best_epoch, best_psnr, color='black', marker='*', s=150, label='Best RMSE Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Validation PSNR')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 6)
    plt.plot(df_log['epoch'], df_log['val_mae'], marker='o', color='brown', label='MAE')
    plt.scatter(best_epoch, best_mae, color='black', marker='*', s=150, label='Best RMSE Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.grid(True)
    plt.legend()
    plt.suptitle(f"Training & Validation Metrics - {experiment_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{experiment_name}_metrics_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 指标变化曲线已保存至: {save_path}")

def plot_final_density(pred, gt, output_path, experiment_name=""):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    # 基础有效过滤
    valid_mask = np.isfinite(gt) & np.isfinite(pred)

    # 额外排除精确0或0.5
    epsilon = 1e-6
    if cfg.data.normalize_range:
        exact_fill = np.abs(gt - 0.5) < epsilon
    else:
        exact_fill = np.abs(gt) < epsilon

    final_valid = valid_mask & ~exact_fill

    pred = pred[final_valid]
    gt = gt[final_valid]

    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(gt, pred, gridsize=100, cmap='viridis', bins='log', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label('log10(Count)')
    max_val = max(gt.max(), pred.max())
    min_val = min(gt.min(), pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    r2 = 1 - (np.sum((gt - pred) ** 2) / np.sum((gt - np.mean(gt)) ** 2)) if np.sum((gt - np.mean(gt)) ** 2) > 0 else 0
    stats_text = f"RMSE = {rmse:.4f}\nR² = {r2:.4f}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f"Final Density Plot - {experiment_name}")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 最终点密度图已保存至：{output_path}")

def train_and_save_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = os.path.join(cfg.data.root_dirs[0], cfg.data.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(output_dir, 'training.log'), level=logging.INFO)
    logging.info("开始训练")

    # 新增：合并多个区域的数据块
    all_blocks = {
        'X': [], 'Y': [], 'landcover': [], 'positions': [], 'target_masks': [],
        'modis_diffs': [], 'landsat_diffs': [], 'modis_months': [], 'landsat_months': [],
        'target_dates': [], 'region_ids': [], 'full_height': None, 'full_width': None
    }
    ref_profile = None

    for idx, root in enumerate(cfg.data.root_dirs):
        print(f"处理区域目录 {idx+1}/{len(cfg.data.root_dirs)}: {root}")
        logging.info(f"处理区域目录: {root}")

        all_modis_paths = sorted(glob.glob(os.path.join(root, cfg.data.modis_pattern)))
        all_landsat_paths = sorted(glob.glob(os.path.join(root, cfg.data.landsat_pattern)))
        landcover_path = os.path.join(root, cfg.data.landcover_file)

        ref_landsat = all_landsat_paths[0] if all_landsat_paths else None
        valid_modis_paths = filter_valid_images(all_modis_paths, ref_path=ref_landsat, threshold=cfg.data.valid_threshold)
        valid_landsat_paths = filter_valid_images(all_landsat_paths, ref_path=ref_landsat, threshold=cfg.data.valid_threshold)

        logging.info(f"有效MODIS影像: {len(valid_modis_paths)}, 有效Landsat影像: {len(valid_landsat_paths)}")

        if not valid_landsat_paths or not valid_modis_paths:
            print(f"警告: 区域 {root} 没有有效影像，跳过")
            logging.warning(f"区域 {root} 没有有效影像，跳过")
            continue

        if ref_profile is None:
            _, ref_profile, _ = load_raster(valid_landsat_paths[0])

        print(f"\n=== 为区域 {root} 创建数据块 ===")
        logging.info(f"为区域 {root} 创建数据块")

        blocks = create_fusion_blocks(
            valid_modis_paths, valid_landsat_paths, landcover_path,
            block_size=cfg.data.block_size, overlap=cfg.data.overlap, augment=cfg.data.data_augment
        )

        if all_blocks['full_height'] is None:
            all_blocks['full_height'] = blocks['full_height']
            all_blocks['full_width'] = blocks['full_width']
        elif (all_blocks['full_height'] != blocks['full_height'] or all_blocks['full_width'] != blocks['full_width']):
            raise ValueError(f"区域 {root} 尺寸 ({blocks['full_height']}, {blocks['full_width']}) 与之前区域不匹配")

        n_blocks = len(blocks['X'])
        all_blocks['region_ids'].extend([idx] * n_blocks)

        for key in ['X', 'Y', 'landcover', 'positions', 'target_masks', 'modis_diffs',
                    'landsat_diffs', 'modis_months', 'landsat_months', 'target_dates']:
            all_blocks[key].extend(blocks[key])

    if not all_blocks['X']:
        raise ValueError("所有区域都没有有效数据块")

    if cfg.data.normalize_range:
        print("=== 对所有合并块应用NDVI归一化 ===")
        logging.info("应用NDVI归一化")
        for i in range(len(all_blocks['X'])):
            all_blocks['X'][i] = normalize_ndvi(all_blocks['X'][i])
            all_blocks['Y'][i] = normalize_ndvi(all_blocks['Y'][i])

    modis_bands = load_raster(valid_modis_paths[0])[0].shape[0]
    landsat_bands = load_raster(valid_landsat_paths[0])[0].shape[0]

    print(f"\n=== 检测到 {modis_bands} MODIS bands 和 {landsat_bands} Landsat bands ===")
    logging.info(f"MODIS bands: {modis_bands}, Landsat bands: {landsat_bands}")

    # 获取所有独特目标日期
    unique_target_dates = sorted(set(all_blocks['target_dates']))
    print(f"所有独特目标日期: {unique_target_dates}")
    logging.info(f"所有独特目标日期: {unique_target_dates}")

    experiments = []
    experiments.append({
        "name": "Lc-transformer-logistic",
        "use_attention": True,
        "use_transformer": True,
        "use_landcover": True,
        "use_logistic": True,
        "mc_dropout": True
    })

    print("实验配置:", experiments)
    logging.info(f"实验配置: {experiments}")

    all_results = []

    for exp_config in experiments:
        try:
            print(f"\n=== Training experiment: {exp_config['name']} ===")
            logging.info(f"开始实验: {exp_config['name']}")

            exp_date_results = []  # 每个日期的 K折结果

            # 对每个日期独立训练
            for target_date in unique_target_dates:
                print(f"\n=== 训练目标日期 {target_date} ===")
                logging.info(f"训练目标日期 {target_date}")

                # 收集该日期的所有块
                date_indices = [i for i, d in enumerate(all_blocks['target_dates']) if d == target_date]
                if len(date_indices) == 0:
                    print(f"日期 {target_date} 没有块，跳过")
                    continue

                date_blocks = {k: [all_blocks[k][i] for i in date_indices] for k in all_blocks if isinstance(all_blocks[k], list)}
                for k in ['full_height', 'full_width']:
                    if k in all_blocks:
                        date_blocks[k] = all_blocks[k]

                date_region_ids = [all_blocks['region_ids'][i] for i in date_indices]

                # 在该日期的块上进行 5 折划分
                indices = np.arange(len(date_region_ids))

                if cfg.data.spatial_split and len(set(date_region_ids)) >= cfg.data.k_folds:
                    skf = StratifiedKFold(n_splits=cfg.data.k_folds, shuffle=True, random_state=42)
                    splits = skf.split(indices, date_region_ids)
                else:
                    kf = KFold(n_splits=min(cfg.data.k_folds, len(indices)), shuffle=True, random_state=42)
                    splits = kf.split(indices)

                date_fold_results = []
                date_models_dict = {}

                for fold, (train_idx, val_idx) in enumerate(splits):
                    print(f"  Date {target_date} Fold {fold + 1}/{cfg.data.k_folds}")
                    logging.info(f"Date {target_date} Fold {fold + 1}/{cfg.data.k_folds}")

                    train_date_blocks = {k: [date_blocks[k][i] for i in train_idx] for k in date_blocks if isinstance(date_blocks[k], list)}
                    val_date_blocks = {k: [date_blocks[k][i] for i in val_idx] for k in date_blocks if isinstance(date_blocks[k], list)}

                    for k in ['full_height', 'full_width']:
                        if k in date_blocks:
                            train_date_blocks[k] = date_blocks[k]
                            val_date_blocks[k] = date_blocks[k]

                    train_dataset = PairedFusionDataset(train_date_blocks, [date_region_ids[i] for i in train_idx], split='train')
                    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, collate_fn=fusion_collate_fn)

                    val_dataset = PairedFusionDataset(val_date_blocks, [date_region_ids[i] for i in val_idx], split='val')
                    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=fusion_collate_fn)

                    model = MultiModalFusionNet(
                        modis_bands=modis_bands,
                        landsat_bands=landsat_bands,
                        max_modis_per_target=cfg.data.max_modis_per_target,
                        max_landsat_per_target=cfg.data.max_landsat_per_target,
                        use_attention=exp_config["use_attention"],
                        use_transformer=exp_config["use_transformer"],
                        use_landcover=exp_config["use_landcover"],
                        mc_dropout=exp_config["mc_dropout"],
                        use_multiscale=cfg.model.use_multiscale,
                        adaptive_logistic=cfg.model.adaptive_logistic
                    ).to(device)

                    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

                    if exp_config.get("use_logistic", False):
                        criterion = CombinedLossWithLogistic(
                            alpha=cfg.training.alpha,
                            logistic_weight=cfg.training.logistic_weight_start,
                            sigma=cfg.model.sigma,
                            embed_params=cfg.model.embed_params,
                            label_smoothing=cfg.training.label_smoothing,
                            gradient_weight=0.1,
                            adaptive_logistic=cfg.model.adaptive_logistic
                        ).to(device)
                    else:
                        criterion = CombinedLoss(alpha=cfg.training.alpha).to(device)

                    scaler = torch.amp.GradScaler('cuda') if cfg.training.use_amp else None

                    best_rmse = float('inf')
                    no_improve = 0
                    fold_log_records = []
                    best_state = None

                    # 动态剔除（如果需要，保持你的逻辑，但现在是每个日期独立）
                    all_dates_in_fold = set(train_dataset.target_dates)  # 这里只有一个日期
                    good_dates = all_dates_in_fold.copy()
                    bad_dates_log = []

                    for epoch in range(1, cfg.training.epochs + 1):
                        train_loss = train_epoch(
                            model, train_loader, device, optimizer, criterion,
                            cfg.data.gradient_accum_steps, epoch, cfg.training.epochs,
                            logistic_weight_scheduler, scaler
                        )

                        # 评估训练集指标
                        train_rmse, train_r2, train_ssim, train_psnr, train_mae, _, _ = evaluate(
                            model, train_loader, device, criterion,
                            output_dir=output_dir,
                            experiment_name=f"{exp_config['name']}_date{target_date}_fold{fold+1}_train"
                        )

                        # 评估验证集指标（该日期的 val fold）
                        val_rmse, val_r2, val_ssim, val_psnr, val_mae, all_preds, all_trues = evaluate(
                            model, val_loader, device, criterion,
                            output_dir=output_dir,
                            experiment_name=f"{exp_config['name']}_date{target_date}_fold{fold+1}_val"
                        )

                        print(f"Date {target_date}, Fold {fold+1}, Epoch {epoch}:")
                        print(f"  Train Loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | Train R²: {train_r2:.4f} | "
                              f"Train SSIM: {train_ssim:.4f} | Train PSNR: {train_psnr:.4f} | Train MAE: {train_mae:.4f}")
                        print(f"  Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val SSIM: {val_ssim:.4f} | "
                              f"Val PSNR: {val_psnr:.4f} | Val MAE: {val_mae:.4f}")

                        logging.info(f"Date {target_date}, Fold {fold+1}, Epoch {epoch}:")
                        logging.info(f"  Train Loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | Train R²: {train_r2:.4f} | "
                                     f"Train SSIM: {train_ssim:.4f} | Train PSNR: {train_psnr:.4f} | Train MAE: {train_mae:.4f}")
                        logging.info(f"  Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val SSIM: {val_ssim:.4f} | "
                                     f"Val PSNR: {val_psnr:.4f} | Val MAE: {val_mae:.4f}")

                        fold_log_records.append({
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_rmse": train_rmse,
                            "train_r2": train_r2,
                            "train_ssim": train_ssim,
                            "train_psnr": train_psnr,
                            "train_mae": train_mae,
                            "val_rmse": val_rmse,
                            "val_r2": val_r2,
                            "val_ssim": val_ssim,
                            "val_psnr": val_psnr,
                            "val_mae": val_mae
                        })

                        if val_rmse < best_rmse:
                            best_rmse = val_rmse
                            best_state = model.state_dict()
                            no_improve = 0
                        else:
                            no_improve += 1

                        if no_improve >= cfg.training.patience:
                            print("Early stopping")
                            logging.info("Early stopping")
                            break

                        scheduler.step()

                    plot_metrics_curve(pd.DataFrame(fold_log_records), f"{exp_config['name']}_date{target_date}_fold{fold+1}", output_dir)

                    if best_state is not None:
                        date_models_dict[f"fold{fold+1}"] = best_state
                        torch.save(best_state, os.path.join(output_dir, f"model_{exp_config['name']}_date{target_date}_fold{fold+1}.pth"))

                    # 该 fold 验证集结果
                    date_fold_results.append({
                        "date": target_date,
                        "fold": fold + 1,
                        "best_val_rmse": best_rmse,
                        "best_val_r2": val_r2,  # 用最后 epoch 的 val
                        "best_val_ssim": val_ssim,
                        "best_val_psnr": val_psnr,
                        "best_val_mae": val_mae
                    })

                # 该日期所有 fold 结果
                exp_date_results.append({
                    "date": target_date,
                    "fold_results": date_fold_results,
                    "avg_rmse": np.mean([r["best_val_rmse"] for r in date_fold_results]),
                    "avg_r2": np.mean([r["best_val_r2"] for r in date_fold_results]),
                    # ... 其他平均
                })

                # 保存该日期所有 fold 模型
                torch.save(date_models_dict, os.path.join(output_dir, f"{exp_config['name']}_date{target_date}_all_folds.pth"))

            # experiment 结果
            all_results.append({
                "experiment": exp_config["name"],
                "date_results": exp_date_results
            })

            # 测试部分（保持你的原逻辑，或根据需要调整为每个日期测试）

        except Exception as e:
            print(f"实验 {exp_config['name']} 失败: {str(e)}")
            logging.error(f"实验 {exp_config['name']} 失败: {str(e)}")
            traceback.print_exc()
            continue

    # 保存总结
    df_results = pd.DataFrame(all_results)
    df_results.to_excel(os.path.join(output_dir, "ablation_study_results.xlsx"), index=False)
    print(df_results)
    print(f"✅ Summary report saved to: {os.path.join(output_dir, 'ablation_study_results.xlsx')}")
    logging.info("训练完成")

if __name__ == "__main__":
    try:
        train_and_save_model()
    except Exception as e:
        print(f"训练失败: {str(e)}")
        logging.error(f"训练失败: {str(e)}")
        traceback.print_exc()
