import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    return feat


class ArcfacePredictor:
    def __init__(self, model_name: str, weight: Path, data_path: Path = None, device='cpu'):
        """

        Args:
            model_name: r18, r34, r50, r100, r200, r2060, mbf
            weight:
            data_path:
            device:
        """
        self.model = get_model(model_name, fp16=False)
        self.model.load_state_dict(torch.load(weight))
        self.model.eval()

        self.transform = tt.Compose([
            tt.Resize(112, interpolation=tt.InterpolationMode.BICUBIC),
            tt.CenterCrop(100),
            tt.ToTensor(),
            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.device = device
        self.model.to(self.device)
        if data_path is not None:
            dataset = datasets.ImageFolder(str(data_path), transform=self.transform)
            self.data_loader = DataLoader(
                dataset=dataset,
                batch_size=64,
                num_workers=4,
            )

    @torch.no_grad()
    def __call__(self, image):
        return self.model(image.to(self.device)).cpu().numpy()

    @torch.no_grad()
    def run(self):
        embeddings = []
        for batch, idx in tqdm(self.data_loader, total=len(self.data_loader)):
            embeddings.append(self.model(batch))
        return torch.hstack(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    feat = inference(args.weight, args.network, args.img)
    print(feat)
