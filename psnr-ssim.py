import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# === CONFIGURATION ===
video_path_1 = 'C:/Users/hi/OneDrive/Documents/VSgima/Chapter C/video1.mp4'
video_path_2 = 'C:/Users/hi/OneDrive/Documents/VSgima/Chapter C/video2.mp4'
use_color_ssim = False  # Set to True to use color (RGB) SSIM

# === INIT ===
cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: One of the videos couldn't be opened.")
    exit()

frame_count = 0
psnr_values = []
ssim_values = []
frame_indices = []

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Resize to match dimensions
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # PSNR
    psnr_val = cv2.PSNR(frame1, frame2)

    # SSIM
    if use_color_ssim:
        ssim_val = ssim(frame1, frame2, channel_axis=2, multichannel=True)
    else:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray1, gray2)

    psnr_values.append(psnr_val)
    ssim_values.append(ssim_val)
    frame_indices.append(frame_count)

    # Display side-by-side
    combined = np.hstack((frame1, frame2))
    cv2.putText(combined, f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Comparison: Left = Original | Right = Compressed", combined)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# === Export CSV ===
results_df = pd.DataFrame({
    "Frame": frame_indices,
    "PSNR": psnr_values,
    "SSIM": ssim_values
})
csv_path = 'C:/Users/hi/OneDrive/Documents/VSgima/Chapter C/psnr_ssim_results.csv'
results_df.to_csv(csv_path, index=False)
print(f"\nFrame-by-frame results saved to:\n{csv_path}")

# === Plot graphs ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(frame_indices, psnr_values, label="PSNR", color='blue')
plt.title("PSNR vs Frame")
plt.xlabel("Frame")
plt.ylabel("PSNR (dB)")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(frame_indices, ssim_values, label="SSIM", color='green')
plt.title("SSIM vs Frame")
plt.xlabel("Frame")
plt.ylabel("SSIM")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# === Final Summary ===
if frame_count > 0:
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
else:
    print("No frames to compare.")
