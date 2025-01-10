import numpy as np
import matplotlib.pyplot as plt
import os
from NeuralLib.config import DATASETS_GIB01

# peak detection - binary
results_dir = r'C:\Users\Catia Bastos\dev\results\peak_detection\checkpoints\ada\GRUseq2seq_[32, 64, ' \
              r'64]hid_3l_bidirTrue_lr0.001_drop[0, 0, 0]_dt2024-12-12_00-07-07\predictions '

path_sig = os.path.join(DATASETS_GIB01, 'x', 'test')
path_idx = os.path.join(DATASETS_GIB01, 'y_idx', 'test')
i = 0

file = os.listdir(path_sig)[i]
# file = 'subject6_session10_segment3.npy'
test_signal = np.load(os.path.join(path_sig, file))
test_idx = np.load(os.path.join(path_idx, file))
pred = np.load(os.path.join(results_dir, file))

plt.plot(test_signal, 'k', lw=0.7)
plt.plot(test_idx, test_signal[test_idx], 'o')
plt.plot(pred, test_signal[pred], '*')
plt.title(file)
plt.show()
