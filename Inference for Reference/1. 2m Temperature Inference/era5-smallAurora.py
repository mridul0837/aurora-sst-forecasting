from pathlib import Path
import numpy as np

# Data will be downloaded here.
download_path = Path("/home/mridul01/scratch/data/aurora-small/inference-metric")


download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)



import torch
import xarray as xr

from aurora import Batch, Metadata

static_vars_ds = xr.open_dataset(download_path / "2025-02-01-static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2025-02-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2025-02-01-atmospheric.nc", engine="netcdf4")



# Batch data
batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),      
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None])         
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

# Inference 
from aurora import AuroraSmallPretrained, rollout

model = AuroraSmallPretrained()
model.load_checkpoint()

model.eval()
model = model.to("cuda")

with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]

model = model.to("cpu")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Prediction")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    ax[i, 1].imshow(surf_vars_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("ERA5")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()
plt.savefig("aurora_small_t2m.png")




# --- RMSE and Correlation ---
rmse_list = []
corr_list = []

for i, pred in enumerate(preds):
    # Prediction and truth for 2m temperature
    pred_t2m = pred.surf_vars["2t"][0, 0].numpy()
    true_t2m = surf_vars_ds["t2m"][2 + i].values[:-1, :]

    # RMSE
    rmse = np.sqrt(np.mean((pred_t2m - true_t2m) ** 2))
    rmse_list.append(rmse)

    # Correlation coefficient (flatten arrays)
    corr = np.corrcoef(pred_t2m.flatten(), true_t2m.flatten())[0, 1]
    corr_list.append(corr)

    print(f"Step {i+1}: RMSE = {rmse:.4f}, Corr = {corr:.4f}")

print(f"Average RMSE: {np.mean(rmse_list):.4f}")
print(f"Average Corr: {np.mean(corr_list):.4f}")





