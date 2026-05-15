import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH  = '/home/arothman/vessap/output/output_crop_sample_2_right_norm_bins.nii.gz'
CENS_PATH = '/home/arothman/vessap/output/output_crop_sample_2_right_norm_bins_cens.nii.gz'
RADS_PATH = '/home/arothman/vessap/output/output_crop_sample_2_right_norm_bins_rads.nii.gz'
BIFS_PATH = '/home/arothman/vessap/output/output_crop_sample_2_right_norm_bins_bifs.nii.gz'

MIP_PCTS   = [10, 50, 100]
MASK_ALPHA = 0.6
AXIS       = 0        # 0=Z projection → gives 512×512 for this volume
DPI        = 150
OUT_PATH   = '/home/arothman/vessap/output/ouput_png_preprocessed_sato.png'

# ── Load + transpose XYZ → ZYX ───────────────────────────────────────────────
def load(path):
    print(f'Loading {os.path.basename(path)}...', flush=True)
    img  = nib.load(path)
    data = np.transpose(img.get_fdata().astype(np.float32), (2, 1, 0))
    return data, img.header.get_zooms()

raw_vol, zoom = load(RAW_PATH)
print(f'Raw shape (ZYX): {raw_vol.shape},  voxel spacing: {zoom}')

# Physical aspect ratio of the projected image (row_spacing / col_spacing)
# After ZYX transpose: axis0=Z(sz), axis1=Y(sy), axis2=X(sx)
vox = {'zyx': (zoom[2], zoom[1], zoom[0])}['zyx']
aspect_map = {0: vox[1]/vox[2],   # project Z → image is (Y, X)
              1: vox[0]/vox[2],   # project Y → image is (Z, X)
              2: vox[0]/vox[1]}   # project X → image is (Z, Y)
aspect = aspect_map[AXIS]

features = []
# cmap: colormap name for continuous data, or an RGBA tuple for solid-colour binary overlays
for path, label, cmap, is_binary in [
    (CENS_PATH, 'Centerlines',  (0.0, 1.0, 0.0, MASK_ALPHA),  True),   # solid lime green
    (RADS_PATH, 'Radius',       'plasma',                       False),  # continuous colormap
    (BIFS_PATH, 'Bifurcations', (1.0, 0.0, 1.0, MASK_ALPHA),  True),   # solid magenta
]:
    if os.path.exists(path):
        vol, _ = load(path)
        features.append((vol, label, cmap, is_binary))
    else:
        print(f'Skipping {label} — not found')

# ── Central MIP ───────────────────────────────────────────────────────────────
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

# ── Figure sized to actual image pixels ───────────────────────────────────────
# Get pixel dims of one MIP panel
sample_mip = mip_central(raw_vol, 100)
H, W = sample_mip.shape                          # e.g. 512 × 512
cell_w = W / DPI                                  # inches
cell_h = H * aspect / DPI                         # inches, corrected for voxel spacing

ncols  = len(MIP_PCTS)
CBAR_W = 0.25  # inches reserved for radius colorbar

for feat_vol, label, cmap, is_binary in features:
    fig_w = cell_w * ncols + (CBAR_W if not is_binary else 0)
    fig_h = cell_h

    right_edge = 1 - CBAR_W / fig_w if not is_binary else 1.0

    fig, axes = plt.subplots(1, ncols,
                             figsize=(fig_w, fig_h),
                             dpi=DPI,
                             squeeze=False)
    fig.subplots_adjust(left=0, right=right_edge, top=0.88, bottom=0,
                        wspace=0.02, hspace=0)

    rad_im = None
    nz = feat_vol[feat_vol > 0]
    rad_vmin = np.percentile(nz, 5)  if (not is_binary and nz.size > 0) else 0
    rad_vmax = np.percentile(nz, 95) if (not is_binary and nz.size > 0) else 1

    for col, pct in enumerate(MIP_PCTS):
        ax = axes[0][col]

        raw_mip  = disp(mip_central(raw_vol, pct))
        feat_mip = mip_central(feat_vol, pct)

        ax.imshow(raw_mip, cmap='gray', vmin=0, vmax=1,
                  origin='lower', aspect=aspect,
                  interpolation='nearest')

        if is_binary:
            rgba = np.zeros((*feat_mip.shape, 4), dtype=np.float32)
            rgba[feat_mip > 0] = cmap
            ax.imshow(rgba, origin='lower', aspect=aspect,
                      interpolation='nearest')
        else:
            masked = np.ma.masked_where(feat_mip == 0, feat_mip)
            rad_im = ax.imshow(masked, cmap=cmap, alpha=MASK_ALPHA,
                               vmin=rad_vmin, vmax=rad_vmax,
                               origin='lower', aspect=aspect,
                               interpolation='nearest')

        ax.set_title(f'{pct}% MIP', fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    if rad_im is not None:
        cbar_ax = fig.add_axes([right_edge + 0.005, 0.1, 0.015, 0.78])
        cbar = fig.colorbar(rad_im, cax=cbar_ax)
        cbar.set_label('Vessel radius (mm)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f'Intensity (grey) + {label} — central MIP',
                 fontsize=11, fontweight='bold', y=0.97)

    out = OUT_PATH.replace('.png', f'_{label.lower()}.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Saved: {out}')
