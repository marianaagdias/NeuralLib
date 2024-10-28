import data_preprocessing.gib01 as gib
from config import RESULTS_PEAK_DETECTION
import BaseModel.model_architectures as bm
from utils.test_models import test_peak_detection_test_set
import time
import os
import glob

if __name__ == '__main__':
    run_possibilities = ['all', 'train', 'test']
    run = run_possibilities[0]
    n_features = 1
    hid_dim = 8
    n_layers = 2
    dropout = 0.3
    epochs = 1
    model = bm.GRUseq2seq(n_features=n_features,
                          hid_dim=hid_dim,
                          n_layers=n_layers,
                          dropout=dropout,
                          learning_rate=0.001,
                          bidirectional=True,
                          task='classification',
                          num_classes=1,
                          batch_size=1,
                          gpu_id=None,
                          results_directory=RESULTS_PEAK_DETECTION)

    if run in ['all', 'train']:
        start_time = time.time()
        model.train_model(path_x=gib.X,
                          path_y=gib.Y_BIN,
                          all_samples=False,
                          samples=3,
                          epochs=epochs,
                          patience=2,
                          dataset_name='gib01',
                          trained_for='peak detection',
                          )

    # checkpoints_directory = model.checkpoints()
    # dire=r"C:\Users\Catia Bastos\dev\results\peak_detection\checkpoints\GRUseq2seq_8hid_2l_lr0.001_drop0.3_dt2024-10-28_17-12-50"
    ckpt_file = glob.glob(os.path.join(model.checkpoints_directory, '*.ckpt'))[0]
    print(ckpt_file)

    if run in ['all', 'test']:
        test_peak_detection_test_set(model_checkpoint=ckpt_file,
                                     path_x=gib.X,
                                     path_y=gib.Y_BIN,
                                     n_features=n_features,
                                     hid_dim=hid_dim,
                                     n_layers=n_layers,
                                     dropout=dropout,
                                     threshold=0.5,
                                     all_samples=False,
                                     samples=5)
