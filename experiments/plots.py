import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import data_preprocessing.gib01 as gib
import os

dirs = [os.path.join(gib.X, 'train'), os.path.join(gib.Y_BIN, 'train'), os.path.join(gib.Y_IDX, 'train')]

sub = 7
session = 10
seg = 9
file = f"subject{sub}_session{session}_segment{seg}.npy"
ecg_d = os.path.join(dirs[0], file)
peaks_bin_d = os.path.join(dirs[1], file)
peaks_idx_d = os.path.join(dirs[2], file)

ecg = np.load(ecg_d)
peaks_idx = np.load(peaks_idx_d)
peaks_bin = np.load(peaks_bin_d)

plt.figure()
plt.plot(ecg)
plt.plot(peaks_bin)
plt.plot(peaks_idx, ecg[peaks_idx], 'ro')
plt.show()

print(ecg.shape, peaks_bin.shape)

dir_clean = r'C:\Users\Catia Bastos\PycharmProjects\ECG_Denoiser\data\Y\Y_train'
dir_noisy = r'C:\Users\Catia Bastos\PycharmProjects\ECG_Denoiser\data\X\X_train'

s = '4_1'
c = np.load(os.path.join(dir_clean, s + '.npy'))
n = np.load(os.path.join(dir_noisy, s + '.npy'))

plt.figure()
plt.plot(c)
plt.plot(n)

