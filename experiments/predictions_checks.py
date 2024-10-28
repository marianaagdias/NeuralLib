import numpy as np
import matplotlib.pyplot as plt
import os
from config import DATASETS_GIB01

# peak detection - binary
results_dir = r'C:\Users\Catia Bastos\dev\results\peak_detection\checkpoints\GRUseq2seq_64hid_3l_lr0.001_drop0.3_dt2024-10-18_18-01-26\predictions'
i = 12

path_sig = os.path.join(DATASETS_GIB01, 'x', 'test')
path_idx = os.path.join(DATASETS_GIB01, 'y_idx', 'test')
file = os.listdir(path_sig)[i]
test_signal = np.load(os.path.join(path_sig, file))
test_idx = np.load(os.path.join(path_idx, file))
pred = np.load(os.path.join(results_dir, file))

print(pred.shape)
#pred_ind = pred[:, 1]

plt.plot(test_signal)
plt.plot(pred, test_signal[pred], 'o')
plt.plot(test_idx, test_signal[test_idx], '*')
plt.show()
