import pandas as pd
import librosa
import librosa.display
import os
import os.path as osp
import shutil
import numpy as np
import glob
import random

from tqdm import tqdm
from scipy.io import wavfile as wav
from wavFileHelper import WavFileHelper
wavfilehelper = WavFileHelper()

from pathlib import Path


def copy_data_into_directory(data_directory, target_directory, xlsx_list, label_list):

    print('Copy processing.....')
    for label in label_list:
        os.makedirs(osp.join(target_directory, label), exist_ok=True)

    for data in tqdm(xlsx_list):
        file_name = data['slice_file_name']
        fold = data['fold']
        label = data['class']
        src = osp.join(data_directory, f'fold{fold}', file_name)
        dst = osp.join(target_directory, label, file_name)
        shutil.copy(src, dst)

    print('Processing is fished')


def extract_features(file_name):

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=2048)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled

def feacture_extraction(target_directory, label_list):

    for label in tqdm(label_list):
        for mode in ['train', 'test']:
            for file in glob.glob(osp.join(target_directory, f'{label}_{mode}', '*.wav')):
                #os.remove(file)
                #file_name = file.split(os.sep)[-1].replace('.wav', '')
                feature = extract_features(file)
                #np.save(osp.join(target_directory, f'{label}_{mode}', f'{file_name}.npy'), feature)
    """
    for label in tqdm(label_list):
        for mode in ['train', 'test']:
            for file in glob.glob(osp.join(target_directory, f'{label}_{mode}', '*.npy')):
                feature = np.load(file)
    """

def train_test_split(data_list, ratio):

    assert isinstance(ratio, list) and len(ratio) == 2

    random.shuffle(data_list)

    cut = int(len(data_list) * 0.8)
    train_data = data_list[:cut]
    test_data = data_list[cut:]

    return train_data, test_data


def generate_data(data_list, target_directory):
    for data in data_list:
        file_name = data.split(os.sep)[-1]
        src = data
        dst = osp.join(target_directory, file_name)
        shutil.copy(src, dst)


def data_split(target_directory, data_directory, label_list, ratio=[0.8, 0.2], seed=101):

    os.makedirs(target_directory, exist_ok=True)
    random.seed(seed)
    for label in tqdm(label_list):
        os.makedirs(osp.join(target_directory, f'{label}_train'), exist_ok=True)
        os.makedirs(osp.join(target_directory, f'{label}_test'), exist_ok=True)
        data_list = glob.glob(osp.join(data_directory, label, '*.wav'))
        train_data, test_data = train_test_split(data_list, ratio)
        generate_data(train_data, osp.join(target_directory, f'{label}_train'))
        generate_data(test_data, osp.join(target_directory, f'{label}_test'))


if __name__ == '__main__':
    pass

    xlxs_path = './UrbanSound8K/metadata/UrbanSound8K.csv'
    data_path = './UrbanSound8K/audio'

    label_list = ["air_conditioner", "car_horn", "children_playing",
                  "dog_bark", "drilling", "engine_idling", "gun_shot",
                  "jackhammer", "siren", "street_music"]
    p = Path(xlxs_path)
    assert p.exists()

    #result = pd.read_csv(p).to_dict(orient='records')

    ### Data Copy
    """
    copy_data_into_directory(data_directory=data_path,
                             target_directory='data',
                             xlsx_list=result, label_list=label_list)
    """

    ### Data Split
    #data_split(target_directory='./tt_data', data_directory='./data', label_list=label_list)

    ### Feature Extraction
    feacture_extraction('./tt_data', label_list)



