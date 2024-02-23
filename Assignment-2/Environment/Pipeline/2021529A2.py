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
        self.transform = torchaudio.transforms.MFCC(n_mfcc=20, log_mels=True, melkwargs=dict(n_fft=400, hop_length=160, n_mels=40))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        waveform, sample_rate, label, _, _ = self.dataset[index]
        if waveform.shape[1] < sample_rate:
            waveform = nn.functional.pad(waveform, (0, sample_rate-waveform.shape[1]))
        MFCC = self.transform(waveform).squeeze(0)
        return MFCC, self.MAPPING[label]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(ResidualBlock, self).__init__()

        assert dim == 1 or dim == 2, "dim must be 1 or 2"

        Conv = eval(f"nn.Conv{dim}d")
        BatchNorm = eval(f"nn.BatchNorm{dim}d")

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
        return nn.functional.relu(out + residual)


class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers_1d = nn.Sequential(
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20, stride=2),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20, stride=2),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20, stride=2),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20, stride=2),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20, stride=2),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20),
            ResidualBlock(dim=1, in_channels=20, out_channels=20, stride=2),
            nn.Flatten(),
            nn.Linear(40, 35)
        )

        self.layers_2d = nn.Sequential(
            ResidualBlock(dim=2, in_channels=3, out_channels=3),                  # Block-01:    3 x 32 x 32 ->    3 x 32 x 32
            ResidualBlock(dim=2, in_channels=3, out_channels=16),                 # Block-02:    3 x 32 x 32 ->   16 x 32 x 32
            ResidualBlock(dim=2, in_channels=16, out_channels=16),                # Block-03:   16 x 32 x 32 ->   16 x 32 x 32
            ResidualBlock(dim=2, in_channels=16, out_channels=16),                # Block-04:   16 x 32 x 32 ->   16 x 32 x 32
            ResidualBlock(dim=2, in_channels=16, out_channels=32, stride=2),      # Block-05:   16 x 32 x 32 ->   32 x 16 x 16
            ResidualBlock(dim=2, in_channels=32, out_channels=32),                # Block-06:   32 x 16 x 16 ->   32 x 16 x 16
            ResidualBlock(dim=2, in_channels=32, out_channels=32),                # Block-07:   32 x 16 x 16 ->   32 x 16 x 16
            ResidualBlock(dim=2, in_channels=32, out_channels=64, stride=2),      # Block-08:   32 x 16 x 16 ->   64 x  8 x  8
            ResidualBlock(dim=2, in_channels=64, out_channels=64),                # Block-09:   64 x  8 x  8 ->   64 x  8 x  8
            ResidualBlock(dim=2, in_channels=64, out_channels=64),                # Block-10:   64 x  8 x  8 ->   64 x  8 x  8
            ResidualBlock(dim=2, in_channels=64, out_channels=128, stride=2),     # Block-11:   64 x  8 x  8 ->  128 x  4 x  4
            ResidualBlock(dim=2, in_channels=128, out_channels=128),              # Block-12:  128 x  4 x  4 ->  128 x  4 x  4
            ResidualBlock(dim=2, in_channels=128, out_channels=128),              # Block-13:  128 x  4 x  4 ->  128 x  4 x  4
            ResidualBlock(dim=2, in_channels=128, out_channels=256, stride=2),    # Block-14:  128 x  4 x  4 ->  256 x  2 x  2
            ResidualBlock(dim=2, in_channels=256, out_channels=256),              # Block-15:  256 x  2 x  2 ->  256 x  2 x  2
            ResidualBlock(dim=2, in_channels=256, out_channels=256),              # Block-16:  256 x  2 x  2 ->  256 x  2 x  2
            ResidualBlock(dim=2, in_channels=256, out_channels=512, stride=2),    # Block-17:  256 x  2 x  2 ->  512 x  1 x  1
            ResidualBlock(dim=2, in_channels=512, out_channels=512),              # Block-18:  512 x  1 x  1 ->  512 x  1 x  1
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_1d(x) if x.shape[1] == 20 else self.layers_2d(x)


class VGGBlock(nn.Module):
    def __init__(self, dim: int, num_convs: int, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super(VGGBlock, self).__init__()

        assert dim == 1 or dim == 2, "dim must be 1 or 2"

        Conv = eval(f"nn.Conv{dim}d")
        MaxPool = eval(f"nn.MaxPool{dim}d")

        layers = [Conv(in_channels, out_channels, kernel_size, padding=padding)]
        for _ in range(num_convs - 1):
            layers.append(Conv(out_channels, out_channels, kernel_size, padding=padding))
        layers.append(MaxPool(kernel_size, ceil_mode=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers_1d = nn.Sequential(
            VGGBlock(dim=1, num_convs=2, in_channels=20, out_channels=20, kernel_size=2, padding=1),
            VGGBlock(dim=1, num_convs=2, in_channels=20, out_channels=13, kernel_size=3, padding=1),
            VGGBlock(dim=1, num_convs=3, in_channels=13, out_channels=9, kernel_size=4, padding=2),
            VGGBlock(dim=1, num_convs=3, in_channels=9, out_channels=6, kernel_size=5, padding=3),
            VGGBlock(dim=1, num_convs=3, in_channels=6, out_channels=4, kernel_size=7, padding=4),
            nn.Flatten(),
            nn.Sequential(
                nn.Linear(12, 20),
                nn.Linear(20, 28),
                nn.Linear(28, 35),
            )
        )

        self.layers_2d = nn.Sequential(
            VGGBlock(dim=2, num_convs=2, in_channels=36, out_channels=36, kernel_size=3, padding=1),
            VGGBlock(dim=2, num_convs=2, in_channels=36, out_channels=24, kernel_size=4, padding=1),
            VGGBlock(dim=2, num_convs=3, in_channels=24, out_channels=16, kernel_size=5, padding=2),
            VGGBlock(dim=2, num_convs=3, in_channels=16, out_channels=11, kernel_size=7, padding=3),
            VGGBlock(dim=2, num_convs=3, in_channels=11, out_channels=8, kernel_size=9, padding=4),
            nn.Flatten(),
            nn.Sequential(
                nn.Linear(8, 8),
                nn.Linear(8, 9),
                nn.Linear(9, 10),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 20:
            return self.layers_1d(x)
        else:
            return self.layers_2d(x.repeat(1, 12, 1, 1))


class CNA(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super(CNA, self).__init__()

        assert dim == 1 or dim == 2, "dim must be 1 or 2"

        Conv = eval(f"nn.Conv{dim}d")
        BatchNorm = eval(f"nn.BatchNorm{dim}d")

        self.layers = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InceptionBlock(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_1x1: int, out_5x5a: int, out_5x5b: int) -> None:
        super(InceptionBlock, self).__init__()

        assert dim == 1 or dim == 2, "dim must be 1 or 2"

        self.branch1 = CNA(dim=dim, in_channels=in_channels, out_channels=out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            CNA(dim=dim, in_channels=in_channels, out_channels=out_5x5a, kernel_size=3, padding=2),
            CNA(dim=dim, in_channels=out_5x5a, out_channels=out_5x5a, kernel_size=5, padding=1)
        )
        self.branch3 = nn.Sequential(
            CNA(dim=dim, in_channels=in_channels, out_channels=out_5x5b, kernel_size=3, padding=2),
            CNA(dim=dim, in_channels=out_5x5b, out_channels=out_5x5b, kernel_size=5, padding=1)
        )
        MaxPool = eval(f"nn.MaxPool{dim}d")
        self.branch4 = MaxPool(3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat((branch1, branch2, branch3, branch4), 1)


class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers_1d = nn.Sequential(
            InceptionBlock(dim=1, in_channels=20, out_1x1=1, out_5x5a=1, out_5x5b=1),
            InceptionBlock(dim=1, in_channels=23, out_1x1=1, out_5x5a=1, out_5x5b=1),
            InceptionBlock(dim=1, in_channels=26, out_1x1=1, out_5x5a=1, out_5x5b=2),
            InceptionBlock(dim=1, in_channels=30, out_1x1=2, out_5x5a=3, out_5x5b=5),
            nn.Flatten(),
            nn.Linear(4040, 35)
        )

        self.layers_2d = nn.Sequential(
            InceptionBlock(dim=2, in_channels=3, out_1x1=1, out_5x5a=2, out_5x5b=2),
            InceptionBlock(dim=2, in_channels=8, out_1x1=1, out_5x5a=1, out_5x5b=2),
            InceptionBlock(dim=2, in_channels=12, out_1x1=2, out_5x5a=2, out_5x5b=4),
            InceptionBlock(dim=2, in_channels=20, out_1x1=2, out_5x5a=3, out_5x5b=5),
            nn.Flatten(),
            nn.Linear(30720, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_1d(x) if x.shape[1] == 20 else self.layers_2d(x)


class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers_1d = nn.Sequential(
            ResidualBlock(dim=1, in_channels=20, out_channels=13, stride=3),
            ResidualBlock(dim=1, in_channels=13, out_channels=9, stride=3),
            InceptionBlock(dim=1, in_channels=9, out_1x1=2, out_5x5a=2, out_5x5b=2),
            InceptionBlock(dim=1, in_channels=15, out_1x1=2, out_5x5a=1, out_5x5b=2),
            ResidualBlock(dim=1, in_channels=20, out_channels=13, stride=3),
            InceptionBlock(dim=1, in_channels=13, out_1x1=2, out_5x5a=2, out_5x5b=3),
            ResidualBlock(dim=1, in_channels=20, out_channels=13, stride=3),
            InceptionBlock(dim=1, in_channels=13, out_1x1=2, out_5x5a=2, out_5x5b=3),
            ResidualBlock(dim=1, in_channels=20, out_channels=13, stride=3),
            InceptionBlock(dim=1, in_channels=13, out_1x1=2, out_5x5a=2, out_5x5b=3),
            nn.Flatten(),
            nn.Linear(20, 35)
        )

        self.layers_2d = nn.Sequential(
            ResidualBlock(dim=2, in_channels=3, out_channels=2, stride=2),
            ResidualBlock(dim=2, in_channels=2, out_channels=2, stride=2),
            InceptionBlock(dim=2, in_channels=2, out_1x1=3, out_5x5a=2, out_5x5b=3),
            InceptionBlock(dim=2, in_channels=10, out_1x1=2, out_5x5a=1, out_5x5b=2),
            ResidualBlock(dim=2, in_channels=15, out_channels=10, stride=2),
            InceptionBlock(dim=2, in_channels=10, out_1x1=2, out_5x5a=1, out_5x5b=2),
            ResidualBlock(dim=2, in_channels=15, out_channels=10, stride=2),
            InceptionBlock(dim=2, in_channels=10, out_1x1=2, out_5x5a=1, out_5x5b=2),
            ResidualBlock(dim=2, in_channels=15, out_channels=10, stride=2),
            InceptionBlock(dim=2, in_channels=10, out_1x1=2, out_5x5a=1, out_5x5b=2),
            nn.Flatten(),
            nn.Linear(15, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_1d(x) if x.shape[1] == 20 else self.layers_2d(x)


def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):

    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)

    best_loss = float("inf")
    no_improvement = 0

    # Write your code here
    for epoch in range(EPOCH):
        total = total_loss = accuracy = 0
        for inputs, labels in dataloader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                total_loss += (loss := criterion(outputs, labels)).item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            except Exception:
                continue
            else:
                accuracy += (outputs.argmax(1) == labels).sum().item()
                total += labels.shape[0]
        accuracy /= total
        loss = total_loss / total
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            loss,
            accuracy
        ))

        checkpoint = {
            "epoch": epoch,
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy
        }
        torch.save(checkpoint, "checkpoint.pth")

        if best_loss - loss >= 1e-4:
            best_loss = loss
            no_improvement = 0
        else:
            if (no_improvement := no_improvement+1) >= 5:
                break
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

    best_loss = float("inf")
    no_improvement = 0

    try:
        checkpoint = torch.load("checkpoint.pth")
    except FileNotFoundError:
        pass
    else:
        network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = checkpoint["loss"]

    # Write your code here
    for epoch in range(EPOCH):
        total = total_loss = accuracy = 0
        for inputs, labels in dataloader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                total_loss += (loss := criterion(outputs, labels)).item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            except Exception:
                continue
            else:
                accuracy += (outputs.argmax(1) == labels).sum().item()
                total += labels.shape[0]
        accuracy /= total
        loss = total_loss / total
        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            loss,
            accuracy
        ))

        checkpoint = {
            "epoch": epoch,
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy
        }
        torch.save(checkpoint, "checkpoint.pth")

        if best_loss - loss >= 1e-4:
            best_loss = loss
            no_improvement = 0
        else:
            if (no_improvement := no_improvement+1) >= 5:
                break
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
    network = network.to(device)
    criterion = nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load("checkpoint.pth")
    except FileNotFoundError:
        pass
    else:
        network.load_state_dict(checkpoint["network"])

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

