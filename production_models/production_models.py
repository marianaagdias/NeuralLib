from production_models.base import ProductionModel
from architectures import GRUseq2seq, GRUseq2seqCustom


class PeakDetector(ProductionModel, GRUseq2seq):
    """Specific trained model for peak detection."""
    def __init__(self, weights_path, training_info, checkpoints_directory=None):
        # Initialize both parents
        ProductionModel.__init__(self, model_name="PeakDetector", weights_path=weights_path,
                              training_info=training_info, checkpoints_directory=checkpoints_directory)
        GRUseq2seq.__init__(self, n_features=10, hid_dim=128, n_layers=2, dropout=0.3, learning_rate=0.001)
        # or GRUseq2seqCustom

