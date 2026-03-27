import matplotlib.pyplot as plt
import numpy as np

# =======================
# 数据
# =======================
rate = np.array([1.0, 0.8331, 0.6133, 0.5289, 0.4122, 0.2996, 0.2403, 0.0076])

psnr  = np.array([34.0127, 34.0528, 33.9988, 33.8980, 33.257, 27.23, 24.46, 8.76])
ssim  = np.array([0.962, 0.962, 0.961, 0.960, 0.956, 0.927, 0.903, 0.327])
lpips = np.array([0.0831, 0.0833, 0.0847, 0.086, 0.0913, 0.1245, 0.1543, 0.81])

# =======================
# 排序（必须）
# =======================
idx = np.argsort(rate)
rate = rate[idx]
psnr = psnr[idx]
ssim = ssim[idx]
lpips = lpips[idx]

# =======================
# 可选：去掉极端点（推荐）
# =======================
mask = rate > 0.05
rate_plot = rate[mask]
psnr_plot = psnr[mask]
ssim_plot = ssim[mask]
lpips_plot = lpips[mask]

# =======================
# PSNR
# =======================
plt.figure(figsize=(6,5))
plt.plot(rate_plot, psnr_plot, '-o', linewidth=2)

plt.xscale('log')
plt.xlabel('Rate')
plt.ylabel('PSNR (dB)')
plt.title('Rate-Distortion Curve (PSNR)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('rd_psnr.png', dpi=300)
plt.close()

# =======================
# SSIM
# =======================
plt.figure(figsize=(6,5))
plt.plot(rate_plot, ssim_plot, '-s', linewidth=2)

plt.xscale('log')
plt.xlabel('Rate')
plt.ylabel('SSIM')
plt.title('Rate-Distortion Curve (SSIM)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('rd_ssim.png', dpi=300)
plt.close()

# =======================
# LPIPS
# =======================
plt.figure(figsize=(6,5))
plt.plot(rate_plot, lpips_plot, '-^', linewidth=2)

plt.xscale('log')
plt.xlabel('Rate')
plt.ylabel('LPIPS')
plt.title('Rate-Distortion Curve (LPIPS)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('rd_lpips.png', dpi=300)
plt.close()

print("Saved: rd_psnr.png, rd_ssim.png, rd_lpips.png")