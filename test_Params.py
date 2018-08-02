import DataHandeling
import os
from datetime import datetime

ROOT_SAVE_DIR = os.path.join('.', 'Outputs')

ROOT_DATA_DIR = os.path.join('.', 'Data')


class TestParamsBase(object):
    """
    DO NOT TOUCH OR CHANGE THIS CLASS UNLESS YOU KNOW WHAT YOU ARE DOING
    """

    def __init__(self):  # DO NOT TOUCH THIS METHOD
        self.data_provider_class = DataHandeling.CSVSegReaderRandom  # DO NOT TOUCH

        self.norm = 2 ** 7  # NORMALIZATION OF THE INPUT IMAGE. PLEASE DO NOT TOUCH

        self.data_base_folder = [ROOT_DATA_DIR]  # DO NOT TOUCH

        self.save_log_dir = ROOT_SAVE_DIR

        # Data and Data Provider
        self.root_data_dir = ROOT_DATA_DIR  # THE DIRECTORY OF THE DATA, SET THIS AT THE TOP OF THE FILE

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_log_dir = os.path.join(self.save_log_dir, self.experiment_name, now_string)

        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)


class TestParams(TestParamsBase):
    # Net Parameters
    # net_build_params = {'layers': 2, 'kernel_size': 7, 'kernel_num': 32, 'num_repeat': 3}
    net_build_params = {'layers': 2, 'kernel_size': 3, 'kernel_num': 16, 'num_repeat': 3}

    # Hardware
    use_gpu = False  # IF NO GPU AVAILABE, SET TO FALSE
    gpu_id = 0  # IF MORE THAN 1 GPU IS AVAILABLE, SELECT WHICH ONE
    dry_run = False  # SET TO TRUE IF YOU DO NOT WANT TO SAVE ANY OUTPUTS (GOOD WHEN DEBUGGING)
    profile = False  # SET TO TRUE FOR THROUGHPUT PROFILING

    image_shape = [512, 640]  # crop size of the input image
    data_dir = os.path.join(ROOT_DATA_DIR, 'Test', 'RawImages')
    # load_checkpoint_path = os.path.join("Best Model", "model_604500.ckpt")
    load_checkpoint_path = os.path.join('.', 'Logs', 'fusion_2GPU', '2018-08-01_115406',
                                        'model_10000.ckpt')
    experiment_name = 'fusion_2gpu'
    filename_regexp = r'calibrate2-P01.[0-9]{3}.TIF'
