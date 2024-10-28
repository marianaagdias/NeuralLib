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
    hid_dim = 64
    n_layers = 3
    dropout = 0.3
    epochs = 80
    model = bm.GRUseq2seq(n_features=n_features,
                          hid_dim=hid_dim,
                          n_layers=n_layers,
                          dropout=dropout,
                          learning_rate=0.001,
                          bidirectional=True,
                          task='classification',
                          num_classes=1,
                          batch_size=64,
                          gpu_id=0,
                          results_directory=RESULTS_PEAK_DETECTION)

    if run in ['all', 'train']:
        start_time = time.time()
        model.train_model(path_x=gib.X,
                          path_y=gib.Y_BIN,
                          all_samples=True,
                          epochs=epochs,
                          patience=20,
                          dataset_name='gib01',
                          trained_for='peak detection',
                          enable_tensorboard=True)
        checkpoints_directory = model.checkpoints_directory
    else:
        checkpoints_directory = os.path.join(RESULTS_PEAK_DETECTION, 'checkpoints',
                                             'GRUseq2seq_64hid_3l_lr0.001_drop0.3_dt2024-10-18_18-01-26')
    print(checkpoints_directory)
    print(os.listdir(checkpoints_directory))
    print(glob.glob(os.path.join(checkpoints_directory, '*.ckpt')))
    ckpt_file = glob.glob(os.path.join(checkpoints_directory, '*.ckpt'))[0]

    if run in ['all', 'test']:
        test_peak_detection_test_set(model_checkpoint=ckpt_file,
                                     path_x=gib.X,
                                     path_y=gib.Y_BIN,
                                     n_features=n_features,
                                     hid_dim=hid_dim,
                                     n_layers=n_layers,
                                     dropout=dropout,
                                     threshold=0.5,
                                     all_samples=True)

    # 2nd model
    # model2 = bm.GRUseq2seq(n_features=n_features,
    #                        hid_dim=128,
    #                        n_layers=2,
    #                        dropout=dropout,
    #                        learning_rate=0.001,
    #                        bidirectional=True,
    #                        task='classification',
    #                        num_classes=1,
    #                        batch_size=32,
    #                        gpu_id=0,
    #                        results_directory=RESULTS_PEAK_DETECTION)
    # start_time = time.time()
    # model2.train_model(path_x=gib.X,
    #                    path_y=gib.Y_BIN,
    #                    all_samples=True,
    #                    epochs=epochs,
    #                    patience=20,
    #                    dataset_name='gib01',
    #                    pretrained_checkpoint=None)
    #
    # checkpoints_directory = model2.checkpoints()
    # ckpt_file2 = glob.glob(os.path.join(checkpoints_directory, '*.ckpt'))[0]
    # test_peak_detection_test_set(model_checkpoint=ckpt_file2,
    #                              path_x=gib.X,
    #                              path_y=gib.Y_BIN,
    #                              n_features=n_features,
    #                              hid_dim=128,
    #                              n_layers=2,
    #                              dropout=dropout,
    #                              threshold=0.5,
    #                              all_samples=True)
