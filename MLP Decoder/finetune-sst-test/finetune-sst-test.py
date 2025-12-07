import numpy as np
import h5py
import xarray as xr
import pickle
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torchvision

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from aurora.batch import Batch, Metadata
from aurora.model.aurora_lite import AuroraLite
from aurora.model.decoder_lite import MLPDecoderLite


# Data path
data_path = Path(f"/scratch/{os.environ['USER']}/data/finetune-data-2020-2024")
data_path = data_path.expanduser()

# Setup
device = torch.device("cuda")

# Load static data once
static_ds = xr.open_dataset(data_path / "static.nc")
static_ds = static_ds.sel(latitude=static_ds.latitude[:720])

# Build file lists
train_surf_files = []
train_atmos_files = []

val_surf_files = []
val_atmos_files = []

# Training: 2020-2021
for year in [2020, 2021]:
    for month in range(1, 13):
        m = f"{month:02d}"
        surf_file = data_path / f"era5_surface_{year}_{m}.nc"
        atmos_file = data_path / f"era5_atmospheric_{year}_{m}.nc"
        
        if surf_file.exists() and atmos_file.exists():
            train_surf_files.append(surf_file)
            train_atmos_files.append(atmos_file)
        else:
            print(f"Warning: Missing training files for {year}-{m}")

# Validation: 2022
for year in [2022]:
    for month in range(1, 13):
        m = f"{month:02d}"
        surf_file = data_path / f"era5_surface_{year}_{m}.nc"
        atmos_file = data_path / f"era5_atmospheric_{year}_{m}.nc"
        
        if surf_file.exists() and atmos_file.exists():
            val_surf_files.append(surf_file)
            val_atmos_files.append(atmos_file)
        else:
            print(f"Warning: Missing validation files for {year}-{m}")

print(f"Training files: {len(train_surf_files)} months")
print(f"Validation files: {len(val_surf_files)} months")


# After building the file lists, add:
train_surf_files = train_surf_files[:1]  # Only first month
train_atmos_files = train_atmos_files[:1]

val_surf_files = val_surf_files[:1]  # Only first month
val_atmos_files = val_atmos_files[:1]

# Models
modelAurora = AuroraLite(
    use_lora=False,
    autocast=True,
    surf_vars=("2t", "10u", "10v", "msl"),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("z", "u", "v", "t", "q"),
)
modelAurora.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
modelAurora = modelAurora.to(device)
modelAurora.eval()

modelDecoder = MLPDecoderLite(
    surf_vars_new=["sst"],
    patch_size=modelAurora.decoder.patch_size,
    embed_dim=2 * modelAurora.encoder.embed_dim,
    hidden_dims=[512, 512, 256],
)
checkpoint = torch.load("../aurora-lite-decoder/lite-decoder.ckpt", map_location="cpu")
modelDecoder.load_state_dict(checkpoint, strict=False)
modelDecoder = modelDecoder.to(device)

# Optimizer
opt = torch.optim.AdamW(modelDecoder.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, 
    mode='min',
    factor=0.5,
    patience=2    
)

# TensorBoard
log_dir=f"/scratch/{os.environ['USER']}/runs/sst_finetune_test"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
global_step = 0

# Best model tracking
best_val_loss = float('inf')
ckpt_dir = Path(f"/scratch/{os.environ['USER']}/checkpoints/sst_finetune_test")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Gradient accumulation settings
accumulation_steps = 22
history = 2

# Masked loss function for SST
def masked_loss(pred, target, mask, loss_fn=F.l1_loss):
    """Compute loss only on valid ocean pixels"""
    valid_pred = pred[mask]
    valid_target = target[mask]
    return loss_fn(valid_pred, valid_target)

print("\nStarting training with gradient accumulation...")
print(f"Accumulation steps: {accumulation_steps}")
print(f"History window: {history}")

# Training loop
for epoch in range(1):
    
    # ========== TRAINING ==========
    modelDecoder.train()
    train_loss = 0.0
    train_rmse = 0.0
    train_mae = 0.0
    train_r2 = 0.0
    
    opt.zero_grad()
    sample_count = 0
    batch_metrics = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}
    
    # Iterate through training files
    for file_idx, (surf_file, atmos_file) in enumerate(zip(train_surf_files, train_atmos_files)):
        
        # Load monthly datasets
        surf = xr.open_dataset(surf_file).isel(latitude=slice(0, 720))
        atmos = xr.open_dataset(atmos_file).isel(latitude=slice(0, 720))
        
        time_dim = 'valid_time'
        n_times = surf.sizes[time_dim]
        
        # Iterate through time steps (starting after history window)
        valid_samples = n_times - history
        usable_samples = (valid_samples // 22) * 22
        end_index = history + usable_samples

        for t in range(history, end_index):
            
            # Extract history slice
            surf_hist = surf.isel({time_dim: slice(t-history, t)})
            atmos_hist = atmos.isel({time_dim: slice(t-history, t)})
            
            # Extract target at time t - SST from surface variables
            target = torch.from_numpy(
                surf["sst"].isel({time_dim: t}).values
            ).float().unsqueeze(0).to(device)  # Add batch dim
            
            # Create ocean mask (True where SST is valid, i.e., ocean pixels)
            ocean_mask = ~torch.isnan(target)
            
            # Create Batch object
            batch = Batch(
                surf_vars={
                    "2t": torch.from_numpy(surf_hist["t2m"].values[:2]).unsqueeze(0),
                    "10u": torch.from_numpy(surf_hist["u10"].values[:2]).unsqueeze(0),
                    "10v": torch.from_numpy(surf_hist["v10"].values[:2]).unsqueeze(0),
                    "msl": torch.from_numpy(surf_hist["msl"].values[:2]).unsqueeze(0),
                },
                static_vars={
                    "z":   torch.from_numpy(static_ds["z"].values[0]),
                    "slt": torch.from_numpy(static_ds["slt"].values[0]),
                    "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
                },
                atmos_vars={
                    "t": torch.from_numpy(atmos_hist["t"].values[:2]).unsqueeze(0),
                    "u": torch.from_numpy(atmos_hist["u"].values[:2]).unsqueeze(0),
                    "v": torch.from_numpy(atmos_hist["v"].values[:2]).unsqueeze(0),
                    "q": torch.from_numpy(atmos_hist["q"].values[:2]).unsqueeze(0),
                    "z": torch.from_numpy(atmos_hist["z"].values[:2]).unsqueeze(0),
                },
                metadata=Metadata(
                    lat=torch.from_numpy(surf_hist.latitude.values),
                    lon=torch.from_numpy(surf_hist.longitude.values),        
                    time=(surf_hist[time_dim].values.astype("datetime64[s]").tolist()[1],),
                    atmos_levels=tuple(int(level) for level in atmos_hist.pressure_level.values),
                )
            )
            
            
            # Forward Aurora encoder (frozen)
            with torch.inference_mode():
                _, latent = modelAurora.forward(batch)
                
            latent_decoder = latent.detach().clone()
            
            # Forward decoder
            preds = modelDecoder(latent_decoder, batch.metadata.lat, batch.metadata.lon)
            pred_sst = preds["sst"].squeeze(1)
            
            # Compute loss (scaled by accumulation steps, only on ocean pixels)
            loss_value = masked_loss(pred_sst, target, ocean_mask, F.l1_loss) / accumulation_steps
            
            # Backprop (gradients accumulate)
            loss_value.backward()
            
            # Accumulate metrics (only on ocean pixels)
            with torch.no_grad():
                valid_pred = pred_sst[ocean_mask]
                valid_target = target[ocean_mask]
                
                mae = F.l1_loss(valid_pred, valid_target)
                mse = F.mse_loss(valid_pred, valid_target)
                rmse = torch.sqrt(mse)
                
                ss_res = torch.sum((valid_target - valid_pred) ** 2)
                ss_tot = torch.sum((valid_target - valid_target.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot
                
                batch_metrics["loss"] += loss_value.item() * accumulation_steps
                batch_metrics["mae"] += mae.item()
                batch_metrics["rmse"] += rmse.item()
                batch_metrics["r2"] += r2.item()
            
            sample_count += 1
            
            # Update weights after accumulation_steps samples
            if sample_count % accumulation_steps == 0:
                
                # Gradient monitoring
                total_norm = 0.0
                for p in modelDecoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar("train/gradient_norm_before_clip", total_norm, global_step)
                
                # Gradient clipping
                grad_norm_after = torch.nn.utils.clip_grad_norm_(modelDecoder.parameters(), max_norm=1.0)
                writer.add_scalar("train/gradient_norm_after_clip", grad_norm_after.item(), global_step)
                
                # Optimizer step
                opt.step()
                opt.zero_grad()
                
                # Log averaged metrics for this batch
                for key in batch_metrics:
                    avg_value = batch_metrics[key] / accumulation_steps
                    writer.add_scalar(f"train/{key}_step", avg_value, global_step)
                    
                    # Accumulate for epoch average
                    if key == "loss":
                        train_loss += avg_value
                    elif key == "mae":
                        train_mae += avg_value
                    elif key == "rmse":
                        train_rmse += avg_value
                    elif key == "r2":
                        train_r2 += avg_value
                
                # Log prediction statistics (only ocean pixels)
                writer.add_scalar("train/pred_mean", valid_pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", valid_pred.std().item(), global_step)
                writer.add_scalar("train/target_mean", valid_target.mean().item(), global_step)
                writer.add_scalar("train/target_std", valid_target.std().item(), global_step)
                
                # Correlation
                correlation = torch.corrcoef(torch.stack([valid_pred, valid_target]))[0, 1]
                writer.add_scalar("train/correlation", correlation.item(), global_step)
                
                # Ocean coverage
                ocean_fraction = ocean_mask.float().mean()
                writer.add_scalar("train/ocean_fraction", ocean_fraction.item(), global_step)
                
                # Visualizations (every 100 steps)
                if global_step % 100 == 0:
                    # For visualization, replace NaN with 0
                    pred_vis = pred_sst[0].clone()
                    target_vis = target[0].clone()
                    pred_vis[~ocean_mask[0]] = 0
                    target_vis[~ocean_mask[0]] = 0
                    
                    writer.add_image("train/pred_sst", pred_vis.unsqueeze(0), global_step, dataformats="CHW")
                    writer.add_image("train/target_sst", target_vis.unsqueeze(0), global_step, dataformats="CHW")
                    
                    diff_vis = (pred_sst[0] - target[0]).clone()
                    diff_vis[~ocean_mask[0]] = 0
                    writer.add_image("train/diff_sst", diff_vis.unsqueeze(0), global_step, dataformats="CHW")
                
                # Reset batch metrics
                batch_metrics = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}
                global_step += 1
                
                if global_step % 100 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | Samples processed: {sample_count}")
        
        # Close monthly datasets
        surf.close()
        atmos.close()
    
    # Compute epoch averages
    num_batches = sample_count // accumulation_steps
    train_loss /= num_batches
    train_mae /= num_batches
    train_rmse /= num_batches
    train_r2 /= num_batches
    
    # ========== VALIDATION ==========
    modelDecoder.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_rmse = 0.0
    val_r2 = 0.0
    val_sample_count = 0
    
    with torch.no_grad():
        for file_idx, (surf_file, atmos_file) in enumerate(zip(val_surf_files, val_atmos_files)):
            
            # Load monthly datasets
            surf = xr.open_dataset(surf_file).isel(latitude=slice(0, 720))
            atmos = xr.open_dataset(atmos_file).isel(latitude=slice(0, 720))
            
            time_dim = 'valid_time'
            n_times = surf.sizes[time_dim]
            
            # Iterate through time steps
            for t in range(history, n_times):
                
                # Extract history slice
                surf_hist = surf.isel({time_dim: slice(t-history, t)})
                atmos_hist = atmos.isel({time_dim: slice(t-history, t)})
                
                # Extract target at time t
                target = torch.from_numpy(
                    surf["sst"].isel({time_dim: t}).values
                ).float().unsqueeze(0).to(device)
                
                # Create ocean mask
                ocean_mask = ~torch.isnan(target)
                
                # Create Batch object
                batch = Batch(
                    surf_vars={
                        "2t": torch.from_numpy(surf_hist["t2m"].values[:2]).unsqueeze(0),
                        "10u": torch.from_numpy(surf_hist["u10"].values[:2]).unsqueeze(0),
                        "10v": torch.from_numpy(surf_hist["v10"].values[:2]).unsqueeze(0),
                        "msl": torch.from_numpy(surf_hist["msl"].values[:2]).unsqueeze(0),
                    },
                    static_vars={
                        "z":   torch.from_numpy(static_ds["z"].values[0]),
                        "slt": torch.from_numpy(static_ds["slt"].values[0]),
                        "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
                    },
                    atmos_vars={
                        "t": torch.from_numpy(atmos_hist["t"].values[:2]).unsqueeze(0),
                        "u": torch.from_numpy(atmos_hist["u"].values[:2]).unsqueeze(0),
                        "v": torch.from_numpy(atmos_hist["v"].values[:2]).unsqueeze(0),
                        "q": torch.from_numpy(atmos_hist["q"].values[:2]).unsqueeze(0),
                        "z": torch.from_numpy(atmos_hist["z"].values[:2]).unsqueeze(0),
                    },
                    metadata=Metadata(
                        lat=torch.from_numpy(surf_hist.latitude.values),
                        lon=torch.from_numpy(surf_hist.longitude.values),        
                        time=(surf_hist[time_dim].values.astype("datetime64[s]").tolist()[1],),
                        atmos_levels=tuple(int(level) for level in atmos_hist.pressure_level.values),
                    )
                )
                
                
                # Forward Aurora encoder
                with torch.inference_mode():
                    _, latent = modelAurora.forward(batch)
                    
                latent_decoder = latent.detach().clone()
                
                # Forward decoder
                preds = modelDecoder(latent_decoder, batch.metadata.lat, batch.metadata.lon)
                pred_sst = preds["sst"].squeeze(1)
                
                # Compute metrics (only on ocean pixels)
                valid_pred = pred_sst[ocean_mask]
                valid_target = target[ocean_mask]
                
                mae = F.l1_loss(valid_pred, valid_target)
                mse = F.mse_loss(valid_pred, valid_target)
                rmse = torch.sqrt(mse)
                
                ss_res = torch.sum((valid_target - valid_pred) ** 2)
                ss_tot = torch.sum((valid_target - valid_target.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot
                
                val_loss += mae.item()
                val_mae += mae.item()
                val_rmse += rmse.item()
                val_r2 += r2.item()
                val_sample_count += 1
                
                # Log first validation sample of epoch
                if val_sample_count == 1:
                    # For visualization, replace NaN with 0
                    pred_vis = pred_sst[0].clone()
                    target_vis = target[0].clone()
                    pred_vis[~ocean_mask[0]] = 0
                    target_vis[~ocean_mask[0]] = 0
                    
                    writer.add_image("val/pred_sst", pred_vis.unsqueeze(0), epoch, dataformats="CHW")
                    writer.add_image("val/target_sst", target_vis.unsqueeze(0), epoch, dataformats="CHW")
                    
                    diff_vis = (pred_sst[0] - target[0]).clone()
                    diff_vis[~ocean_mask[0]] = 0
                    writer.add_image("val/diff_sst", diff_vis.unsqueeze(0), epoch, dataformats="CHW")
            
            # Close monthly datasets
            surf.close()
            atmos.close()
    
    # Compute validation averages
    val_loss /= val_sample_count
    val_mae /= val_sample_count
    val_rmse /= val_sample_count
    val_r2 /= val_sample_count

    scheduler.step(val_loss)  # Adjust lr based on val_loss
    
    # ========== LOGGING ==========
    print(f"Epoch {epoch+1}/{10} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
    
    writer.add_scalar("train/loss_epoch", train_loss, epoch)
    writer.add_scalar("train/mae_epoch", train_mae, epoch)
    writer.add_scalar("train/rmse_epoch", train_rmse, epoch)
    writer.add_scalar("train/r2_epoch", train_r2, epoch)
    
    writer.add_scalar("val/loss_epoch", val_loss, epoch)
    writer.add_scalar("val/mae_epoch", val_mae, epoch)
    writer.add_scalar("val/rmse_epoch", val_rmse, epoch)
    writer.add_scalar("val/r2_epoch", val_r2, epoch)
    
    writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            modelDecoder.state_dict(),
            ckpt_dir / "sst_decoder_best.ckpt"
        )
        print(f"  → New best model saved! (val_loss: {val_loss:.4f})")
    
    # Save regular checkpoint
    torch.save(
        modelDecoder.state_dict(),
        ckpt_dir / f"sst_decoder_epoch{epoch}.ckpt"
    )
    
    writer.flush()

writer.close()
print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
print(f"Checkpoints saved to: {ckpt_dir}")