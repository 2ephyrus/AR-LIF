import torch
import torch.distributed
import random
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps

import math
from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional, Dict
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms.transforms import RandomErasing
class SNNAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

        self.cutout = RandomErasing(p=1, scale=(0.05, 0.33), ratio=(0.2, 5))


    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
            "Cutout": (torch.linspace(1.0, 30.0, num_bins), True)
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        if op_name == "Cutout":
            return self.cutout(img)
        else:
            return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)

def _apply_op(img: Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)

    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class DVSAugment:
    def __init__(self):
        pass

    class Cutout:
        """Randomly mask out one or more patches from an image.
        Args:
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        """
        def __init__(self, ratio):
            self.ratio = ratio

        def __call__(self, img):
            h = img.size(1)
            w = img.size(2)
            lenth_h = int(self.ratio * h)
            lenth_w = int(self.ratio * w)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - lenth_h // 2, 0, h)
            y2 = np.clip(y + lenth_h // 2, 0, h)
            x1 = np.clip(x - lenth_w // 2, 0, w)
            x2 = np.clip(x + lenth_w // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
            return img

    class Roll:
        def __init__(self, off):
            self.off = off

        def __call__(self, img):
            l = int(img.size(1) * self.off)
            off1 = random.randint(-l, l)
            off2 = random.randint(-l, l)
            return torch.roll(img, shifts=(off1, off2), dims=(1, 2))

    def function_nda(self, data, M=1, N=2):
        c = 15 * N
        rotate = transforms.RandomRotation(degrees=c)
        e = N / 6
        cutout = self.Cutout(ratio=e)
        a = N / 6
        roll = self.Roll(off=a)

        transforms_list = [roll, rotate, cutout]
        sampled_ops = np.random.choice(transforms_list, M)
        for op in sampled_ops:
            data = op(data)
        return data

    def trans(self, data):
        flip = random.random() > 0.5
        if flip:
            data = torch.flip(data, dims=(2, ))
        data = self.function_nda(data)
        return data

    def __call__(self, img):
        return self.trans(img)


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# code from https://github.com/yhhhli/SNN_Calibration/blob/master/data/autoaugment.py


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2,
                 fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int_),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10}

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128, 128, 128, 128)),
                                   rot).convert(img.mode)

        func = {
            "shearX":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0
                                                  ), Image.BICUBIC, fillcolor=fillcolor),
            "shearY":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0
                                                  ), Image.BICUBIC, fillcolor=fillcolor),
            "translateX":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, 0, magnitude * img.size[0] * random.choice([
                                                     -1, 1]), 0, 1, 0), fillcolor=fillcolor),
            "translateY":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, 0, 0, 0, 1, magnitude * img.size[1] * random.
                                                  choice([-1, 1])), fillcolor=fillcolor),
            "rotate":
            lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color":
            lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([
                -1, 1])),
            "posterize":
            lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize":
            lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast":
            lambda img, magnitude: ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice(
                [-1, 1])),
            "sharpness":
            lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.
                                                                       choice([-1, 1])),
            "brightness":
            lambda img, magnitude: ImageEnhance.Brightness(img).enhance(1 + magnitude * random.
                                                                        choice([-1, 1])),
            "autocontrast":
            lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize":
            lambda img, magnitude: ImageOps.equalize(img),
            "invert":
            lambda img, magnitude: ImageOps.invert(img)}

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class CIFAR10Policy(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)


# class ImageNetPolicy(object):
#     """ Randomly choose one of the best 24 Sub-policies on ImageNet.
#
#         Example:
#         >>> policy = ImageNetPolicy()
#         >>> transformed = policy(image)
#
#         Example as a PyTorch Transform:
#         >>> transform=transforms.Compose([
#         >>>     transforms.Resize(256),
#         >>>     ImageNetPolicy(),
#         >>>     transforms.ToTensor()])
#     """
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.policies = [
#             SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
#             SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
#             SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
#             SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
#             SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
#             SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
#             SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
#             SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
#             SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
#             SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
#             SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
#             SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
#             SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
#             SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
#             SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
#             SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
#             SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
#             SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
#             SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
#             SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
#             SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)]
#
#     def __call__(self, img):
#         policy_idx = random.randint(0, len(self.policies) - 1)
#         return self.policies[policy_idx](img)
