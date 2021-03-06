import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import easydict
import numpy as np
import torch
from tqdm import tqdm

from backbones import get_model
from eval import verification


def make_bins(source_directory_path: str, destination_folder: str, filename: str, batch_size=30000):
    paths = list(Path(source_directory_path).rglob('*.jpg'))
    images = [np.frombuffer(path.open('rb').read(), np.uint8) for path in paths]

    # there are 2770 people with just one photo.
    identities_dict = defaultdict(list)
    for idx, path in enumerate(paths):
        identities_dict[path.parent.name].append(idx)

    multiple_photos_identities_dict = {
        key: val for key, val in identities_dict.items() if len(val) > 1
    }
    multiple_photos_unique_identities = list(multiple_photos_identities_dict.keys())
    multiple_photos = [path for path in paths if
                       path.parent.name in multiple_photos_unique_identities]
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
        with open(os.path.join(destination_folder, f'{filename}_{i // batch_size:02d}.bin'),
                  'wb') as f:
            pickle.dump((bins[i:i + batch_size], issame_list[i // 2:(i + batch_size) // 2]), f)
        result.append(os.path.join(destination_folder, f'{filename}_{i // batch_size:02d}.bin'))

    return result


def make_negative_bin(source_directory_path: str, destination_folder: str, filename: str):
    paths = list(Path(source_directory_path).rglob('*.jpg'))
    images = [np.frombuffer(path.open('rb').read(), np.uint8) for path in paths]

    identities_dict = defaultdict(list)
    for idx, path in enumerate(paths):
        identities_dict[path.parent.name].append(idx)

    single_photo_identities_dict = {
        key: val for key, val in identities_dict.items() if len(val) == 1
    }
    single_photo_unique_identities = list(single_photo_identities_dict.keys())
    single_photo = [path for path in paths if path.parent.name in single_photo_unique_identities]
    bins = []
    issame_list = [False] * (len(single_photo) // 2)

    for i in range(len(single_photo) // 2):
        anchor = single_photo_identities_dict[single_photo_unique_identities[2 * i]][0]
        negative = single_photo_identities_dict[single_photo_unique_identities[2 * i + 1]][0]
        bins.extend([images[anchor], images[negative]])

    negative_output = os.path.join(destination_folder, f'{filename}_negative.bin')
    with open(negative_output, 'wb') as f:
        pickle.dump((bins, issame_list), f)
    return negative_output


def evaluate(verification_set_path, image_size):
    data_set = verification.load_bin(verification_set_path, image_size)
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
    # backbone.to('cpu')
    verification.test(data_set, backbone, batch_size, folds)


def run_make_bin(split, shall_make_bins=True, shall_make_negative_bins=True):
    source = f'/home/agata/projects/face_search/storage/datasets/CASIA-WebFaces_warped/normalized_images/{split}'
    destination_folder = f'/home/agata/projects/face_search/storage/datasets/CASIA-WebFaces_warped/normalized_mxnet'
    destinations = []
    if shall_make_bins:
        destinations.extend(make_bins(source, destination_folder, split))
    if shall_make_negative_bins:
        destinations.append(make_negative_bin(source, destination_folder, split))
    return destinations


def test_with_agedb():
    path = '/home/agata/projects/images/faces_webface_112x112/agedb_30.bin'
    image_size = (112, 112)
    evaluate(path, image_size)


if __name__ == '__main__':
    run_make_bin('test', shall_make_bins=False)
