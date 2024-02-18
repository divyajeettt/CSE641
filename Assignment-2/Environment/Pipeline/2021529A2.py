import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader, random_split

"""
Write Code for Downloading Image and Audio Dataset Here
"""
# Image Downloader
image_dataset_downloader = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False,                        # change download to True
    transform=torchvision.transforms.ToTensor()
)

image_train_set, image_test_set, image_val_set = random_split(
    image_dataset_downloader, [0.7, 0.2, 0.1]
)

# Audio Downloader
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root="data", download=False, url="speech_commands_v0.02",        # change download to True
)

audio_train_set, audio_test_set, audio_val_set = random_split(
    audio_dataset_downloader, [0.7, 0.2, 0.1]
)

class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        pass
        """
        Write your code here
        """
        self.dataset = eval(f"image_{self.datasplit}_set")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        return self.dataset[index]


class AudioDataset(Dataset):
    MAPPING: dict[str, int] = {
        "backward": 0, "bed": 1, "bird": 2, "cat": 3, "dog": 4, "down": 5, "eight": 6, "five": 7,
        "follow": 8, "forward": 9, "four": 10, "go": 11, "happy": 12, "house": 13, "learn": 14,
        "left": 15, "marvin": 16, "nine": 17, "no": 18, "off": 19, "on": 20, "one": 21, "right": 22,
        "seven": 23, "sheila": 24, "six": 25, "stop": 26, "three": 27, "tree": 28, "two": 29,
        "up": 30, "visual": 31, "wow": 32, "yes": 33, "zero": 34
    }

    INV_MAPPING: dict[int, str] = {label: word for word, label in MAPPING.items()}

    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        pass
        """
        Write your code here
        """
        self.dataset = eval(f"audio_{self.datasplit}_set")
        self.fixed_len = 16000

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16384)
        waveform = resampler(waveform)
        if waveform.shape[0] < self.fixed_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.fixed_len - waveform.size(1)))
        elif waveform.shape[0] > self.fixed_len:
            waveform = waveform[:, :self.fixed_len]
        return waveform.clone(), AudioDataset.MAPPING[label]


class ResnetBlock(nn.Module):
    """
    Represents a Residual-Network block built using the given specifications
    """

    def __init__(self, **kwargs) -> None:
        super(ResnetBlock, self).__init__()

        self.dim = kwargs.get("dim", 0)
        if self.dim != 1 and self.dim != 2:
            raise Exception("dim must be 1 or 2")

        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]
        stride = kwargs.get("stride", 1)
        padding = kwargs.get("padding", 1)

        Conv = eval(f"nn.Conv{self.dim}d")
        BatchNorm = eval(f"nn.BatchNorm{self.dim}d")
        ReLU = nn.ReLU

        self.layers = nn.Sequential(
            Conv(in_channels, out_channels, 3, stride=stride, padding=padding),
            BatchNorm(out_channels),
            ReLU(inplace=True),
            Conv(out_channels, out_channels, 3, stride=1, padding=padding),
            BatchNorm(out_channels)
        )

        self.residual_conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            BatchNorm(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        if out.shape != residual.shape:
            residual = self.residual_conv(residual)
        return nn.functional.relu(out + residual)


class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

        self.layers1d = nn.Sequential(
            ResnetBlock(dim=1, in_channels=1, out_channels=2, stride=1, padding=1),        # Block-01:      1 x 16000 ->    2 x 16000
            ResnetBlock(dim=1, in_channels=2, out_channels=4, stride=2),                   # Block-02:      2 x 16000 ->    4 x  8000
            ResnetBlock(dim=1, in_channels=4, out_channels=8, stride=1, padding=1),        # Block-03:      4 x  8000 ->    8 x  8000
            ResnetBlock(dim=1, in_channels=8, out_channels=8, stride=2),                   # Block-04:      8 x  8000 ->    8 x  4000
            ResnetBlock(dim=1, in_channels=8, out_channels=16, stride=2),                  # Block-05:      8 x  4000 ->   16 x  2000
            ResnetBlock(dim=1, in_channels=16, out_channels=16, stride=1, padding=1),      # Block-06:     16 x  2000 ->   16 x  2000
            ResnetBlock(dim=1, in_channels=16, out_channels=32, stride=2),                 # Block-07:     16 x  2000 ->   32 x  1000
            ResnetBlock(dim=1, in_channels=32, out_channels=32, stride=1, padding=1),      # Block-08:     32 x  1000 ->   32 x  1000
            ResnetBlock(dim=1, in_channels=32, out_channels=64, stride=2),                 # Block-09:     32 x  1000 ->   64 x   500
            ResnetBlock(dim=1, in_channels=64, out_channels=64, stride=1, padding=1),      # Block-10:     64 x   500 ->   64 x   500
            ResnetBlock(dim=1, in_channels=64, out_channels=128, stride=2),                # Block-11:     64 x   500 ->  128 x   250
            ResnetBlock(dim=1, in_channels=128, out_channels=128, stride=1, padding=1),    # Block-12:    128 x   250 ->  128 x   250
            ResnetBlock(dim=1, in_channels=128, out_channels=256, stride=2),               # Block-13:    128 x   250 ->  256 x   125
            ResnetBlock(dim=1, in_channels=256, out_channels=256, stride=1, padding=1),    # Block-14:    256 x   125 ->  256 x   125
            ResnetBlock(dim=1, in_channels=256, out_channels=512, stride=2),               # Block-15:    256 x   125 ->  512 x    63
            ResnetBlock(dim=1, in_channels=512, out_channels=512, stride=2),               # Block-16:    512 x    63 ->  512 x    32
            ResnetBlock(dim=1, in_channels=512, out_channels=1024, stride=2),              # Block-17:    512 x    32 -> 1024 x    16
            ResnetBlock(dim=1, in_channels=1024, out_channels=1024, stride=2),             # Block-18:   1024 x    16 -> 1024 x     8
            nn.Flatten(),
            nn.Linear(1024*8, 35)
        )

        self.layers2d = nn.Sequential(
            ResnetBlock(dim=2, in_channels=3, out_channels=3, stride=1, padding=1),        # Block-01:    3 x 32 x 32 ->    3 x 32 x 32
            ResnetBlock(dim=2, in_channels=3, out_channels=16, stride=1, padding=1),       # Block-02:    3 x 32 x 32 ->   16 x 32 x 32
            ResnetBlock(dim=2, in_channels=16, out_channels=16, stride=1, padding=1),      # Block-03:   16 x 32 x 32 ->   16 x 32 x 32
            ResnetBlock(dim=2, in_channels=16, out_channels=16, stride=1, padding=1),      # Block-04:   16 x 32 x 32 ->   16 x 32 x 32
            ResnetBlock(dim=2, in_channels=16, out_channels=32, stride=2),                 # Block-05:   16 x 32 x 32 ->   32 x 16 x 16
            ResnetBlock(dim=2, in_channels=32, out_channels=32, stride=1, padding=1),      # Block-06:   32 x 16 x 16 ->   32 x 16 x 16
            ResnetBlock(dim=2, in_channels=32, out_channels=32, stride=1, padding=1),      # Block-07:   32 x 16 x 16 ->   32 x 16 x 16
            ResnetBlock(dim=2, in_channels=32, out_channels=64, stride=2),                 # Block-08:   32 x 16 x 16 ->   64 x  8 x  8
            ResnetBlock(dim=2, in_channels=64, out_channels=64, stride=1, padding=1),      # Block-09:   64 x  8 x  8 ->   64 x  8 x  8
            ResnetBlock(dim=2, in_channels=64, out_channels=64, stride=1, padding=1),      # Block-10:   64 x  8 x  8 ->   64 x  8 x  8
            ResnetBlock(dim=2, in_channels=64, out_channels=128, stride=2),                # Block-11:   64 x  8 x  8 ->  128 x  4 x  4
            ResnetBlock(dim=2, in_channels=128, out_channels=128, stride=1, padding=1),    # Block-12:  128 x  4 x  4 ->  128 x  4 x  4
            ResnetBlock(dim=2, in_channels=128, out_channels=128, stride=1, padding=1),    # Block-13:  128 x  4 x  4 ->  128 x  4 x  4
            ResnetBlock(dim=2, in_channels=128, out_channels=256, stride=2),               # Block-14:  128 x  4 x  4 ->  256 x  2 x  2
            ResnetBlock(dim=2, in_channels=256, out_channels=256, stride=1, padding=1),    # Block-15:  256 x  2 x  2 ->  256 x  2 x  2
            ResnetBlock(dim=2, in_channels=256, out_channels=256, stride=1, padding=1),    # Block-16:  256 x  2 x  2 ->  256 x  2 x  2
            ResnetBlock(dim=2, in_channels=256, out_channels=512, stride=2),               # Block-17:  256 x  2 x  2 ->  512 x  1 x  1
            ResnetBlock(dim=2, in_channels=512, out_channels=512, stride=1, padding=1),    # Block-18:  512 x  1 x  1 ->  512 x  1 x  1
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return self.layers1d(x)
        else:
            return self.layers2d(x)


class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):

    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")

    network = network.to(device)

    # Write your code here
    for epoch in range(EPOCH):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        pass
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """

def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):

    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")

    network = network.to(device)

    # Write your code here
    for epoch in range(EPOCH):
        pass
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def evaluator(dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):

    # Write your code here
    for epoch in range(EPOCH):
        pass
    """
    Only use this print statement to print your loss, accuracy
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """

