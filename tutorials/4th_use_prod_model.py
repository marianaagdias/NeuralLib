from NeuralLib.config import DATASETS_PTB_DENOISER
import NeuralLib.model_hub as mh
import os
import numpy as np
import matplotlib.pyplot as plt

# TODO: fazer este tutorial com o denoiser para fazer referencia ao paper (no notebook)

# path_sig = os.path.join(DATASETS_PTB_DENOISER, 'x', 'test')
# path_clean_sig = os.path.join(DATASETS_PTB_DENOISER, 'y', 'test')
# i = 0
# file = os.listdir(path_sig)[i]
# test_signal = np.load(os.path.join(path_sig, file))
# idx = np.load(os.path.join(path_clean_sig, file))

mh.list_production_models()

denoiser = mh.ProductionModel(model_name="ECGDenoiserNL")
# and then:
predicted_signal = denoiser.predict(
    X=test_signal,
    gpu_id=None
)

# check results
plt.plot(test_signal, 'k', lw=0.7)
plt.plot(predicted_signal)
plt.title(file)
plt.show()

