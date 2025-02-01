from NeuralLib.config import DATASETS_GIB01
from NeuralLib.production_models import ProductionModel
from NeuralLib.architectures import post_process_peaks_binary
import os
import numpy as np
import matplotlib.pyplot as plt

path_sig = os.path.join(DATASETS_GIB01, 'x', 'test')
# path_idx = os.path.join(DATASETS_GIB01, 'y_idx', 'test')
i = 0
file = os.listdir(path_sig)[i]
test_signal = np.load(os.path.join(path_sig, file))

ProductionModel.list_collection_models()

peak_detector = ProductionModel(model_name="ECGPeakDetector")
# and then:
predicted_peaks = peak_detector.predict(
    X=test_signal,
    gpu_id=None,
    post_process_fn=post_process_peaks_binary,
    threshold=0.5,
    filter_peaks=True
)

# check results
print("Predicted Peaks:", predicted_peaks)
plt.plot(test_signal, 'k', lw=0.7)
# plt.plot(test_idx, test_signal[test_idx], 'o')
plt.plot(predicted_peaks, test_signal[predicted_peaks], '*')
plt.title(file)
plt.show()

print(predicted_peaks)
