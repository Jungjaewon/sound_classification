import os.path as osp
import glob
import numpy as np
import torch

from torch.utils import data
from torchvision import transforms as T


class DataSet(data.Dataset):

    def __init__(self, config, mode, transform):

        self.data_dir = config['TRAINING_CONFIG']['DATA_DIR']
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        self.data_list = list()
        self.mode = mode
        self.transform = transform

        for idx, label in enumerate(self.label_list):
            label_data = glob.glob(osp.join(self.data_dir, f'{label}_{mode}', '*.npy'))
            label_data = [[x, idx] for x in label_data]
            self.data_list.extend(label_data)


    def __getitem__(self, index):
        data_path, label = self.data_list[index]
        data = np.load(data_path)
        data = np.expand_dims(data, axis=0)
        data = torch.from_numpy(data)
        #return self.transform(data), label
        return data, label

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config, mode):

    transform = list()
    transform.append(T.Normalize(mean=(0.5), std=(0.5)))
    transform = T.Compose(transform)

    dataset = DataSet(config, mode, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'] if mode == 'train' else 1,
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
