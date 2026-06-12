import dataclasses

import numpy as np
import tqdm
from termcolor import cprint
import pickle
import os
from SubFunctions.CreateDataset import CreateDataset
from SubFunctions.GetPreprocessing import Preprocessing
from SubFunctions.Utils import *
from sklearn.preprocessing import LabelEncoder
from SubFunctions.GetFeatures import FeatureExtraction


@dataclasses.dataclass
class ReadDataset:
    # Initializes the ReadDataset class with a dataset identifier and a flag to determine whether to extract or load features
    def __init__(self, dataset: int, exec: bool):
        self.dataset = dataset
        self.exec = exec

    # Reads and processes data from the specified dataset, either extracting features and labels or loading them from a saved file
    def read_data(self):
        if self.exec:
            # Extracts features and labels from the dataset if the exec flag is True
            cprint(f"[⁉️] Extracting the Extracted Features and Labels from Dataset {self.dataset} ", color='grey',
                   on_color='on_white')

            Features = []
            Labels = []

            if self.dataset == 1:
                # Processes data from Dataset 1, extracting features and labels for various activities
                path = "Dataset\\MobiFall_Dataset_v2.0"

                dataset, categories = CreateDataset(path).create()

                # Save first 100 signals
                save_dir = "Test_data/MobiFall_Dataset/"
                os.makedirs(save_dir, exist_ok=True)

                num_samples_to_save = min(100, len(dataset))

                for idx in range(num_samples_to_save):
                    signal_data = dataset[idx].values

                    np.save(
                        os.path.join(save_dir, f"sample{idx + 1}.npy"),
                        signal_data
                    )

                cprint(
                    f"[✓] Saved {num_samples_to_save} signals in {save_dir}",
                    color="green"
                )

                codes = {'STD': 'Standing',
                         'WAL': 'Walking',
                         'JOG': 'Jogging',
                         'JUM': 'Jumping',
                         'STU': 'Stairs up',
                         'STN': 'Stairs down',
                         'SCH': 'Sit chair',
                         'CSI': 'Car-step in',
                         'CSO': 'Car-step out',
                         'FOL': 'Forward-lying',
                         'FKL': 'Front-knees-lying',
                         'BSC': 'Back-sitting-chair',
                         'SDL': 'Sideward-lying'}

                # for i in tqdm.tqdm(range(len(dataset)), desc='Extracting Features'):
                #     # Iterates through each sample in Dataset 1, preprocessing signals and extracting features
                #     data = dataset[i].values
                #     ch_feat = []
                #     for ch in range(data.shape[1]):
                #         signal = data[:, ch]


            else:

                folders = os.listdir(
                    "Dataset\\KFall_Dataset\\Time_series_DB1\\KFall Dataset\\KFall Dataset\\sensor_data")
                for folder in tqdm.tqdm(folders):
                    # Reads label data for each folder in Dataset 2
                    label_data = pd.read_excel(
                        f"Dataset\\KFall Dataset\\KFall Dataset\\label_data\\{folder}_label.xlsx")

                    Labels_ = create_labels(label_data)

                    save_dir = "Saved_Signals_KFall"
                    os.makedirs(save_dir, exist_ok=True)

                    sample_count = 0
                    max_samples = 100

                    for i in range(len(Labels_)):
                        # Iterates through each label entry, extracting and preprocessing sensor data
                        d = Labels_[i]

                        name = folder.replace("A", "")
                        task_number = str(d.get('Task_Number', '')).zfill(2)
                        Rep_Number = str(d.get('Rep_Number', '')).zfill(2)
                        data_path = f"DATASET\\Dataset2\\KFall Dataset\\KFall Dataset\\sensor_data\\{folder}\\{name}T{task_number}R{Rep_Number}.csv"
                        if not os.path.exists(data_path):
                            continue

                        sensor_data = pd.read_csv(data_path)
                        sensor_data['Label'] = 0
                        sensor_data.loc[
                            (sensor_data['FrameCounter'] >= d['Start_Sample']) &
                            (sensor_data['FrameCounter'] <= d['End_Sample']),
                            'Label'
                        ] = d['Class_Label']

                        sensor_data = sensor_data.drop(['TimeStamp(s)', 'FrameCounter'], axis=1)

                        data, labels = create_windows(sensor_data, 50, 10)

                        for k in range(len(data)):

                            # Save first 100 windows
                            if sample_count < max_samples:
                                np.save(
                                    os.path.join(
                                        save_dir,
                                        f"sample{sample_count + 1}.npy"
                                    ),
                                    data[k]
                                )

                                sample_count += 1

                            # ch_feat = []
                            #
                            # for ch in range(data[k].shape[1]):
                            #     signal = data[k][:, ch]


ReadDataset(2, exec=True).read_data()
