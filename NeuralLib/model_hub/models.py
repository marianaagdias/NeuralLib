from NeuralLib.model_hub.production_model import ProductionModel
from NeuralLib.architectures import post_process_peaks_binary


class ECGPeakDetector(ProductionModel):
    """
    Production model for Peak Detection, extending the ProductionModel class.
    Includes additional methods for signal-specific inference and analysis.
    """

    def __init__(self):
        super().__init__(model_name="ECGPeakDetector")

    def detect_peaks(self, signal, gpu_id=None, threshold=0.5, filter_peaks=True):
        processed_output = self.predict(signal,
                                        post_process_fn=post_process_peaks_binary,
                                        gpu_id=gpu_id,
                                        threshold=threshold,  # **post_process_kwargs
                                        filter_peaks=filter_peaks)  # "**post_process_kwargs
        return processed_output

