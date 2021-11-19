import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as tt

from arcface_torch.backbones import get_model


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
    def __init__(self, model_name: str, weight: Path, device: str = 'cpu', **kwargs: Any) -> None:
        """

        Args:
            model_name: r18, r34, r50, r100, r200, r2060, mbf
            weight:
            device:
        """
        self.model = get_model(model_name, fp16=False, **kwargs)
        self.model.load_state_dict(torch.load(weight))
        self.model.eval()

        self.transform = tt.Compose([
            tt.ToPILImage(),
            tt.Resize(112, interpolation=tt.InterpolationMode.BICUBIC),
            tt.CenterCrop(100),
            tt.ToTensor(),
            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, image):
        return self.model(self.transform(image).unsqueeze(0).to(self.device)).cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    feat = inference(args.weight, args.network, args.img)
    print(feat)
