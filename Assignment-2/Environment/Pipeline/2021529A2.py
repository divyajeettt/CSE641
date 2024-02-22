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
    root="data", train=True, download=True,                        # change download to True
    transform=torchvision.transforms.ToTensor()
)

image_train_set, image_test_set, image_val_set = random_split(
    image_dataset_downloader, [0.7, 0.2, 0.1]
)

# Audio Downloader
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root="data", download=True, url="speech_commands_v0.02",        # change download to True
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
        self.dataset = eval(f"audio_{self.datasplit}_set")
        self.length = 10000
        self.sample_rate = 8000
        # self.transform = torchaudio.transforms.MelSpectrogram(self.sample_rate)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        old = waveform.shape
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
        waveform = resampler(waveform)
        if waveform.shape[1] < self.length:
            waveform = nn.functional.pad(waveform, (0, self.length - waveform.shape[1]))
        elif waveform.shape[1] > self.length:
            waveform = waveform[:, :self.length]
        # mel_spectrogram = self.transform(waveform).reshape(81, 128)
        # return mel_spectrogram, AudioDataset.MAPPING[label]
        return waveform, AudioDataset.MAPPING[label]


class CNA(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super(CNA, self).__init__()

        self.dim = dim
        assert self.dim == 1 or self.dim == 2, "dim must be 1 or 2"

        Conv = eval(f"nn.Conv{self.dim}d")
        BatchNorm = eval(f"nn.BatchNorm{self.dim}d")

        self.layers = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InceptionBlock(nn.Module):
    def __init__(self, dim: int, channels: int) -> None:
        super(InceptionBlock, self).__init__()

        self.dim = dim
        assert self.dim == 1 or self.dim == 2, "dim must be 1 or 2"

        self.branch1 = CNA(dim=self.dim, in_channels=channels, out_channels=channels, kernel_size=1)
        self.branch2 = nn.Sequential(
            CNA(dim=self.dim, in_channels=channels, out_channels=channels, kernel_size=3, padding=2),
            CNA(dim=self.dim, in_channels=channels, out_channels=channels, kernel_size=5, padding=1)
        )
        self.branch3 = nn.Sequential(
            CNA(dim=self.dim, in_channels=channels, out_channels=channels, kernel_size=3, padding=2),
            CNA(dim=self.dim, in_channels=channels, out_channels=channels, kernel_size=5, padding=1)
        )
        MaxPool = eval(f"nn.MaxPool{self.dim}d")
        self.branch4 = MaxPool(3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("x", x.shape)
        branch1 = self.branch1(x)
        # print("branch1", branch1.shape)
        branch2 = self.branch2(x)
        # print("branch2", branch2.shape)
        branch3 = self.branch3(x)
        # print("branch3", branch3.shape)
        branch4 = self.branch4(x)
        # print("branch4", branch4.shape)
        return torch.cat((branch1, branch2, branch3, branch4), 1)


class ResnetBlock(nn.Module):
    """
    Represents a Residual-Network block built using the given specifications
    """

    def __init__(self, dim: int, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(ResnetBlock, self).__init__()

        self.dim = dim
        assert self.dim == 1 or self.dim == 2, "dim must be 1 or 2"

        Conv = eval(f"nn.Conv{self.dim}d")
        BatchNorm = eval(f"nn.BatchNorm{self.dim}d")

        self.layers = nn.Sequential(
            Conv(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
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
        # print(out.shape)
        return nn.functional.relu(out + residual)


class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # process raw resampled audio
        self.layers_1d = nn.Sequential(
            ResnetBlock(dim=1, in_channels=1, out_channels=2, stride=1),        # Block-01:      1 x 16000 ->    2 x 16000
            ResnetBlock(dim=1, in_channels=2, out_channels=2, stride=1),        # Block-02:      2 x 16000 ->    2 x 16000
            ResnetBlock(dim=1, in_channels=2, out_channels=4, stride=3),        # Block-03:      2 x 16000 ->    4 x  5334
            ResnetBlock(dim=1, in_channels=4, out_channels=4, stride=1),        # Block-04:      4 x  5334 ->    4 x  5334
            ResnetBlock(dim=1, in_channels=4, out_channels=8, stride=2),        # Block-05:      4 x  5334 ->    8 x  2667
            ResnetBlock(dim=1, in_channels=8, out_channels=8, stride=1),        # Block-06:      8 x  2667 ->    8 x  2667
            ResnetBlock(dim=1, in_channels=8, out_channels=16, stride=2),       # Block-07:      8 x  2667 ->   16 x  1334
            ResnetBlock(dim=1, in_channels=16, out_channels=16, stride=1),      # Block-08:     16 x  1334 ->   16 x  1334
            ResnetBlock(dim=1, in_channels=16, out_channels=32, stride=3),      # Block-09:     16 x  1334 ->   32 x   445
            ResnetBlock(dim=1, in_channels=32, out_channels=32, stride=1),      # Block-10:     32 x   445 ->   32 x   445
            ResnetBlock(dim=1, in_channels=32, out_channels=64, stride=2),      # Block-11:     32 x   445 ->   64 x   223
            ResnetBlock(dim=1, in_channels=64, out_channels=64, stride=3),      # Block-12:     64 x   223 ->   64 x    75
            ResnetBlock(dim=1, in_channels=64, out_channels=128, stride=2),     # Block-13:     64 x    75 ->  128 x    38
            ResnetBlock(dim=1, in_channels=128, out_channels=128, stride=3),    # Block-14:    128 x    38 ->  128 x    13
            ResnetBlock(dim=1, in_channels=128, out_channels=256, stride=2),    # Block-15:    128 x    13 ->  256 x     7
            ResnetBlock(dim=1, in_channels=256, out_channels=256, stride=2),    # Block-16:    256 x     7 ->  256 x     4
            ResnetBlock(dim=1, in_channels=256, out_channels=512, stride=2),    # Block-17:    256 x     4 ->  512 x     2
            ResnetBlock(dim=1, in_channels=512, out_channels=512, stride=2),    # Block-18:    512 x     2 ->  512 x     1
            nn.Flatten(),
            nn.Linear(512, 35)
        )

        (
            # process mel spectrograms of size 128 x 81 -> resized to 81 x 128 (81 channels of length 128)
            # self.layers_1d = nn.Sequential(
            #     ResnetBlock(dim=1, in_channels=81, out_channels=64, stride=1),
            #     ResnetBlock(dim=1, in_channels=64, out_channels=64, stride=1),
            #     ResnetBlock(dim=1, in_channels=64, out_channels=64, stride=1),
            #     ResnetBlock(dim=1, in_channels=64, out_channels=128, stride=2),
            #     ResnetBlock(dim=1, in_channels=128, out_channels=128, stride=1),
            #     ResnetBlock(dim=1, in_channels=128, out_channels=128, stride=1),
            #     ResnetBlock(dim=1, in_channels=128, out_channels=128, stride=2),
            #     ResnetBlock(dim=1, in_channels=128, out_channels=256, stride=1),
            #     ResnetBlock(dim=1, in_channels=256, out_channels=256, stride=2),
            #     ResnetBlock(dim=1, in_channels=256, out_channels=256, stride=1),
            #     ResnetBlock(dim=1, in_channels=256, out_channels=512, stride=1),
            #     ResnetBlock(dim=1, in_channels=512, out_channels=512, stride=1),
            #     ResnetBlock(dim=1, in_channels=512, out_channels=512, stride=1),
            #     ResnetBlock(dim=1, in_channels=512, out_channels=1024, stride=2),
            #     ResnetBlock(dim=1, in_channels=1024, out_channels=1024, stride=1),
            #     ResnetBlock(dim=1, in_channels=1024, out_channels=1024, stride=2),
            #     ResnetBlock(dim=1, in_channels=1024, out_channels=2048, stride=2),
            #     ResnetBlock(dim=1, in_channels=2048, out_channels=2048, stride=2),
            #     nn.Flatten(),
            #     nn.Linear(2048, 35)
            # )
        )

        self.layers_2d = nn.Sequential(
            ResnetBlock(dim=2, in_channels=3, out_channels=3, stride=1),        # Block-01:    3 x 32 x 32 ->    3 x 32 x 32
            ResnetBlock(dim=2, in_channels=3, out_channels=16, stride=1),       # Block-02:    3 x 32 x 32 ->   16 x 32 x 32
            ResnetBlock(dim=2, in_channels=16, out_channels=16, stride=1),      # Block-03:   16 x 32 x 32 ->   16 x 32 x 32
            ResnetBlock(dim=2, in_channels=16, out_channels=16, stride=1),      # Block-04:   16 x 32 x 32 ->   16 x 32 x 32
            ResnetBlock(dim=2, in_channels=16, out_channels=32, stride=2),      # Block-05:   16 x 32 x 32 ->   32 x 16 x 16
            ResnetBlock(dim=2, in_channels=32, out_channels=32, stride=1),      # Block-06:   32 x 16 x 16 ->   32 x 16 x 16
            ResnetBlock(dim=2, in_channels=32, out_channels=32, stride=1),      # Block-07:   32 x 16 x 16 ->   32 x 16 x 16
            ResnetBlock(dim=2, in_channels=32, out_channels=64, stride=2),      # Block-08:   32 x 16 x 16 ->   64 x  8 x  8
            ResnetBlock(dim=2, in_channels=64, out_channels=64, stride=1),      # Block-09:   64 x  8 x  8 ->   64 x  8 x  8
            ResnetBlock(dim=2, in_channels=64, out_channels=64, stride=1),      # Block-10:   64 x  8 x  8 ->   64 x  8 x  8
            ResnetBlock(dim=2, in_channels=64, out_channels=128, stride=2),     # Block-11:   64 x  8 x  8 ->  128 x  4 x  4
            ResnetBlock(dim=2, in_channels=128, out_channels=128, stride=1),    # Block-12:  128 x  4 x  4 ->  128 x  4 x  4
            ResnetBlock(dim=2, in_channels=128, out_channels=128, stride=1),    # Block-13:  128 x  4 x  4 ->  128 x  4 x  4
            ResnetBlock(dim=2, in_channels=128, out_channels=256, stride=2),    # Block-14:  128 x  4 x  4 ->  256 x  2 x  2
            ResnetBlock(dim=2, in_channels=256, out_channels=256, stride=1),    # Block-15:  256 x  2 x  2 ->  256 x  2 x  2
            ResnetBlock(dim=2, in_channels=256, out_channels=256, stride=1),    # Block-16:  256 x  2 x  2 ->  256 x  2 x  2
            ResnetBlock(dim=2, in_channels=256, out_channels=512, stride=2),    # Block-17:  256 x  2 x  2 ->  512 x  1 x  1
            ResnetBlock(dim=2, in_channels=512, out_channels=512, stride=1),    # Block-18:  512 x  1 x  1 ->  512 x  1 x  1
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 81 for mel spectrogram case
        # return self.layers_1d(x) if x.shape[1] == 81 else self.layers_2d(x)
        return self.layers_1d(x) if x.shape[1] == 1 else self.layers_2d(x)


class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers_1d = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(256, 256, 4, stride=1, padding=1),
                nn.Conv1d(256, 256, 4, stride=1, padding=1),
                nn.MaxPool1d(4, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv1d(256, 167, 5, stride=1, padding=1),
                nn.Conv1d(167, 167, 5, stride=1, padding=1),
                nn.MaxPool1d(5, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv1d(167, 109, 7, stride=1, padding=1),
                nn.Conv1d(109, 109, 7, stride=1, padding=1),
                nn.Conv1d(109, 109, 7, stride=1, padding=1),
                nn.MaxPool1d(7, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv1d(109, 71, 9, stride=1, padding=1),
                nn.Conv1d(71, 71, 9, stride=1, padding=1),
                nn.Conv1d(71, 71, 9, stride=1, padding=1),
                nn.MaxPool1d(9, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv1d(71, 47, 12, stride=1, padding=4),
                nn.Conv1d(47, 47, 12, stride=1, padding=4),
                nn.Conv1d(47, 47, 12, stride=1, padding=4),
                nn.MaxPool1d(12, ceil_mode=True)
            ),
            nn.Flatten(),
            nn.Sequential(
                nn.Linear(47, 43),
                nn.Linear(43, 39),
                nn.Linear(39, 35),
            )
        )

        self.layers_2d = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.MaxPool2d(3, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 333, 4, stride=1, padding=1),
                nn.Conv2d(333, 333, 4, stride=1, padding=1),
                nn.MaxPool2d(4, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv2d(333, 217, 5, stride=1, padding=2),
                nn.Conv2d(217, 217, 5, stride=1, padding=2),
                nn.Conv2d(217, 217, 5, stride=1, padding=2),
                nn.MaxPool2d(5, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv2d(217, 142, 7, stride=1, padding=3),
                nn.Conv2d(142, 142, 7, stride=1, padding=3),
                nn.Conv2d(142, 142, 7, stride=1, padding=3),
                nn.MaxPool2d(7, ceil_mode=True)
            ),
            nn.Sequential(
                nn.Conv2d(142, 93, 9, stride=1, padding=4),
                nn.Conv2d(93, 93, 9, stride=1, padding=4),
                nn.Conv2d(93, 93, 9, stride=1, padding=4),
                nn.MaxPool2d(9, ceil_mode=True)
            ),
            nn.Flatten(),
            nn.Sequential(
                nn.Linear(93, 50),
                nn.Linear(50, 25),
                nn.Linear(25, 10),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return self.layers_1d(x.repeat(1, 256, 1))
        else:
            return self.layers_2d(x.repeat(1, 512//3+1, 1, 1)[:, :-1, :, :])


class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers_1d = nn.Sequential(
            InceptionBlock(dim=1, channels=1),
            InceptionBlock(dim=1, channels=1*4),
            InceptionBlock(dim=1, channels=1*4*4),
            InceptionBlock(dim=1, channels=1*4*4*4),
            nn.Flatten(),
            nn.Linear(1*4*4*4*4*16000, 35)
        )

        self.layers_2d = nn.Sequential(
            InceptionBlock(dim=2, channels=3),
            InceptionBlock(dim=2, channels=3*4),
            InceptionBlock(dim=2, channels=3*4*4),
            InceptionBlock(dim=2, channels=3*4*4*4),
            nn.Flatten(),
            nn.Linear(3*4*4*4*4*32*32, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_1d(x) if x.shape[1] == 1 else self.layers_2d(x)


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
    print("in trainer now")
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")

    network = network.to(device)

    # Write your code here
    for epoch in range(EPOCH):
        total = total_loss = accuracy = 0
        for i, (inputs, labels) in enumerate(dataloader):
            # try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                total_loss += (loss := criterion(outputs, labels)).item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # except Exception:
                # continue
            # else:
                accuracy += (outputs.argmax(1) == labels).sum().item()
                total += labels.shape[0]
                print(i, "of", len(dataloader))
        accuracy /= total
        loss = total_loss / total
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            loss,
            accuracy
        ))
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
        total = loss = accuracy = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            loss += criterion(outputs, labels).item()
            accuracy += (outputs.argmax(1) == labels).sum().item()
            total += labels.shape[0]
        accuracy /= total
        loss /= total
        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            loss,
            accuracy
        ))
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):

    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")

    # Write your code here
    total = loss = accuracy = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = network(inputs)
        loss += criterion(outputs, labels).item()
        accuracy += (outputs.argmax(1) == labels).sum().item()
        total += labels.shape[0]
    accuracy /= total
    loss /= total
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """
    Only use this print statement to print your loss, accuracy
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """

