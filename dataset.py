from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import transforms
import skimage.transform as Skimage


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        self.mode = mode
        is_train = mode == Dataset.Mode.TRAIN
        train_transform = transforms.Compose([transforms.Resize(256)])

        self._cifar10 = datasets.CIFAR10(path_to_data_dir, train=is_train, download=True, transform=train_transform)
        print('1')


    def __len__(self) -> int:
        return len(self._cifar10)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == Dataset.Mode.TRAIN:
            return Skimage.resize(self._cifar10.train_data[index], (224, 224)), self._cifar10.train_labels[index]
        else:
            return Skimage.resize(self._cifar10.test_data[index], (224, 224)), self._cifar10.test_labels[index]

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        # TODO: CODE BEGIN
        raise NotImplementedError
        # TODO: CODE END
