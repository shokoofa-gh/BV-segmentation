import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH  = '/home/arothman/vessap/data/output_crop_sample_2_right_norm.nii.gz'
BINS_PATH = '/home/arothman/vessap/output/output_crop_sample_2_right_norm_bins.nii.gz'

MIP_PCTS    = [10, 50, 100]
MASK_COLOUR = (1.0, 0.2, 0.2, 0.5)   # red, 50% alpha
AXIS        = 0
DPI         = 150
OUT_PATH    = '/home/arothman/vessap/output/mask_overlay_vis_normal.png'

# ── Load + transpose XYZ → ZYX ───────────────────────────────────────────────
def load(path):
    print(f'Loading {os.path.basename(path)}...', flush=True)
    img  = nib.load(path)
    data = np.transpose(img.get_fdata().astype(np.float32), (2, 1, 0))
    return data, img.header.get_zooms()

raw_vol,  zoom = load(RAW_PATH)
bins_vol, _    = load(BINS_PATH)
print(f'Shape (ZYX): {raw_vol.shape},  voxel spacing: {zoom}')

vox    = (zoom[2], zoom[1], zoom[0])
aspect = {0: vox[1]/vox[2], 1: vox[0]/vox[2], 2: vox[0]/vox[1]}[AXIS]

# ── MIP + normalisation ───────────────────────────────────────────────────────
def mip_central(vol, pct):
    n    = vol.shape[AXIS]
    half = max(1, int(n * pct / 200))
    mid  = n // 2
    sl   = [slice(None)] * 3
    sl[AXIS] = slice(mid - half, mid + half)
    return np.max(vol[tuple(sl)], axis=AXIS)

def disp(img, lo=1, hi=99):
    lo_v, hi_v = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - lo_v) / (hi_v - lo_v + 1e-8), 0, 1)

# ── Figure: 2 rows (raw / raw+mask), 3 cols (10% / 50% / 100%) ───────────────
sample_mip = mip_central(raw_vol, 100)
H, W       = sample_mip.shape
cell_w     = W / DPI
cell_h     = H * aspect / DPI
nrows, ncols = 2, len(MIP_PCTS)

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(cell_w * ncols, cell_h * nrows),
                         dpi=DPI, squeeze=False)
fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0,
                    wspace=0.02, hspace=0.04)

row_labels = ['Raw', 'Raw + mask']

for col, pct in enumerate(MIP_PCTS):
    raw_mip  = disp(mip_central(raw_vol,  pct))
    bins_mip = mip_central(bins_vol, pct)

    # row 0 — raw only
    ax = axes[0][col]
    ax.imshow(raw_mip, cmap='gray', vmin=0, vmax=1,
              origin='lower', aspect=aspect, interpolation='nearest')
    ax.set_title(f'{pct}% MIP', fontsize=10, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # row 1 — raw + mask overlay
    ax = axes[1][col]
    ax.imshow(raw_mip, cmap='gray', vmin=0, vmax=1,
              origin='lower', aspect=aspect, interpolation='nearest')
    rgba = np.zeros((*bins_mip.shape, 4), dtype=np.float32)
    rgba[bins_mip > 0] = MASK_COLOUR
    ax.imshow(rgba, origin='lower', aspect=aspect, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

for row, label in enumerate(row_labels):
    axes[row][0].set_ylabel(label, fontsize=10, fontweight='bold', labelpad=4)

fig.suptitle('Raw image and segmentation mask — central MIP',
             fontsize=11, fontweight='bold', y=0.97)
plt.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', pad_inches=0)
plt.close()
print(f'Saved: {OUT_PATH}')
