import os
import pickle
from collections import defaultdict
from pathlib import Path
import random

import easydict
import numpy as np
import torch
from tqdm import tqdm

from backbones import get_model
from eval import verification


def make_bin(source_directory_path: str, destination_folder: str, filename: str, batch_size = 30000):
    paths = list(Path(source_directory_path).rglob('*.jpg'))
    images = [np.frombuffer(path.open('rb').read(), np.uint8) for path in paths]

    # there are 2770 people with just one photo.

    identities_dict = defaultdict(list)
    for idx, path in enumerate(paths):
        identities_dict[path.parent.name].append(idx)

    # single_photo = list(filter(lambda path: len(identities_dict[path.parent.name]) == 1, paths))
    # multiple_photos = list(filter(lambda path: len(identities_dict[path.parent.name]) > 1, paths))

    multiple_photos_identities_dict = {
        key: val for key, val in identities_dict.items() if len(val) > 1
    }
    multiple_photos_unique_identities = list(multiple_photos_identities_dict.keys())
    multiple_photos = [path for path in paths if path.parent.name in multiple_photos_unique_identities]
    bins = []
    issame_list = []

    for anchor in tqdm(range(len(multiple_photos))):
        identity = multiple_photos[anchor].parent.name
        positive = random.choice(multiple_photos_identities_dict[identity])
        if positive == anchor:
            positive = (positive + 1) % len(multiple_photos_identities_dict[identity])
        while True:
            negative_identity = random.choice(multiple_photos_unique_identities)
            if negative_identity != identity:
                break
        negative = random.choice(multiple_photos_identities_dict[negative_identity])
        bins.extend([images[anchor], images[positive], images[anchor], images[negative]])
        issame_list.extend([True, False])

    result = []
    for i in range(0, len(bins), batch_size):
        with open(os.path.join(destination_folder, f'{filename}_{i//batch_size:02d}.bin'), 'wb') as f:
            pickle.dump((bins[i:i+batch_size], issame_list[i//2:(i+batch_size)//2]), f)
        result.append(os.path.join(destination_folder, f'{filename}_{i//batch_size:02d}.bin'))
    return result


def blabla(path, image_size):
    data_set = verification.load_bin(path, image_size)
    batch_size = 10
    folds = 10
    cfg = easydict.EasyDict(network='r18', fp16=False, embedding_size=512)
    local_rank = 0
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).to(local_rank)
    backbone_pth = Path(__file__).parents[1] / 'output' / 'webface_arcface_r18' / 'backbone.pth'
    backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
    backbone.eval()
    backbone.to('cpu')
    verification.test(data_set, backbone, batch_size, folds)


def run_make_bin(split):
    source = f'/home/agata/projects/face_search/storage/datasets/CASIA-WebFaces_warped/normalized_images/{split}'
    destination_folder = f'/home/agata/projects/face_search/storage/datasets/CASIA-WebFaces_warped/normalized_mxnet'
    destinations = make_bin(source, destination_folder, split)
    return destinations


def test_with_agedb():
    path = '/home/agata/projects/images/faces_webface_112x112/agedb_30.bin'
    image_size = (112, 112)
    blabla(path, image_size)


if __name__ == '__main__':
    split = 'dev'
    paths = run_make_bin(split)
    for path in paths:
        print(path)
    for path in paths:
        blabla(path, (100, 100))
