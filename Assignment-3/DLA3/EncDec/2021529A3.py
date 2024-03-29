import os
import torch
import random
import torchvision
from EncDec import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity


class AlteredMNIST:
    """
    Represents the given modified MNIST dataset. Due to the unavailability of the
    original mapping for augmentation, we map four augmented images per label to
    one clean image per label. The images are named "Data/X/X_I_L.png":
        - X: {aug=[augmented], clean=[clean]}
        - I: {Index range(0, 60000)}
        - L: {Labels range(10)}
    :attrs:
        - root: The root directory
        - augmented: The list of paths to augmented images
        - clean: Labelwise mapping of clean images
        - mapping: Mapping of augmented images to clean images
        - transform: The preprocessing transformation pipeline
    """

    def __init__(self):
        self.root = os.getcwd()
        self.augmented = [os.path.join(r"Data/aug", image) for image in os.listdir(os.path.join(self.root, r"Data/aug"))]

        self.clean = {str(label): [] for label in range(10)}
        for image in os.listdir(os.path.join(self.root, r"Data/clean")):
            label = image[-5]
            self.clean[label].append(os.path.join(r"Data/clean", image))

        self.mapping = {}
        indices, counts = [0]*10, [0]*10
        for aug in self.augmented:
            label = int(aug[-5])
            self.mapping[aug] = self.clean[str(label)][indices[label]]
            counts[label] += 1
            if counts[label] == 4:
                indices[label] += 1
                counts[label] = 0

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.augmented)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        """
        Returns the augmented and clean image pair with the label of the image
        at the given index.
        """
        aug_path = self.augmented[index]
        label = int(aug_path[-5])
        aug = self.get_pil_image(aug_path)
        clean = self.get_pil_image(self.mapping[aug_path])
        return self.transform(aug), self.transform(clean), torch.tensor(label)

    def get_pil_image(self, path: str) -> "PIL.Image.Image":
        """
        Reads the image at the given path and returns the PIL image.
        """
        return torchvision.transforms.functional.to_pil_image(
            torchvision.io.read_image(os.path.join(self.root, path))
        )


class EncoderBlock(torch.nn.Module):
    """
    Represents a Residual Encoder Block for the AutoEncoder. The block is built with
    the ResNet style specifications:
        - Convolutional Layer with kernel size 3x3
        - Batch Normalization Layer
        - ReLU Activation Layer
        - Convolutional Layer with kernel size 3x3
        - Batch Normalization Layer
        - Residual Connection
    :attrs:
        - layers: The layers of the Encoder Block
        - residual_conv: The convolutional layer for the residual connection
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(EncoderBlock, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder Block.
        """
        residual = x
        out = self.layers(x)
        if out.shape != residual.shape:
            residual = self.residual_conv(x)
        return torch.nn.functional.relu(out + residual)


class Encoder(torch.nn.Module):
    """
    Represents the Encoder for the AutoEncoder. The Encoder consists of 5 Encoder
    Blocks. The encoded logits/embeddings are of shape [batch_size, 128, 2, 2].
    :attrs:
        - layers: The layers of the Encoder
    # TODO: Implement Variational Encoder in the same class
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = torch.nn.Sequential(
            EncoderBlock(in_channels=1, out_channels=8, stride=1),
            EncoderBlock(in_channels=8, out_channels=16, stride=3),
            EncoderBlock(in_channels=16, out_channels=32, stride=3),
            EncoderBlock(in_channels=32, out_channels=64, stride=3),
            EncoderBlock(in_channels=64, out_channels=128, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder.
        """
        return self.layers(x)


class DecoderBlock(torch.nn.Module):
    """
    Represents a Residual Decoder Block for the AutoEncoder. The block is built with
    the ResNet style specifications:
        - Transposed Convolutional Layer with kernel size 3x3
        - Batch Normalization Layer
        - ReLU Activation Layer
        - Transposed Convolutional Layer with kernel size 3x3
        - Batch Normalization Layer
        - Residual Connection
    :attrs:
        - layers: The layers of the Decoder Block
        - residual_conv: The convolutional layer for the residual connection
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DecoderBlock, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder Block.
        """
        residual = x
        out = self.layers(x)
        if out.shape != residual.shape:
            residual = self.residual_conv(x)
        return torch.nn.functional.relu(out + residual)


class Decoder(torch.nn.Module):
    """
    Represents the Decoder for the AutoEncoder. The Decoder consists of 5 Decoder
    Blocks (to approximately undo the encoding). The output images are of shape [batch_size, 1, 28, 28].
    :attrs:
        - layers: The layers of the Decoder
    # TODO: Implement Variational Decoder in the same class
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = torch.nn.Sequential(
            DecoderBlock(in_channels=128, out_channels=64, stride=1),
            DecoderBlock(in_channels=64, out_channels=32, stride=3),
            DecoderBlock(in_channels=32, out_channels=16, stride=3),
            DecoderBlock(in_channels=16, out_channels=8, stride=3),
            DecoderBlock(in_channels=8, out_channels=1, stride=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder.
        """
        return self.layers(z)


class AELossFn(torch.nn.Module):
    """
    Represents the Loss Function for the AutoEncoder. The loss function is a modified
    Mean Squared Error - it mimics the Structural Similarity Index, so we optimize
    directly for SSIM (or at least that's the idea).
    """

    def __init__(self):
        super(AELossFn, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Loss Function.
        """
        MSE = torch.nn.functional.mse_loss(output, target, reduction="none")
        SSIM = torch.clamp((1 - MSE/255), min=0.0, max=1.0)
        return (1 - (SSIM+1)/2).sum()


class VAELossFn(torch.nn.Module):
    """
    Represents the Loss Function for the Variational AutoEncoder. The loss function
    is a combination of the (above-described) SSIM and KL Divergence.
    """

    def __init__(self):
        super(VAELossFn, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Loss Function.
        """
        MSE = torch.nn.functional.mse_loss(output, target, reduction="none")
        SSIM = torch.clamp((1 - MSE/255), min=0.0, max=1.0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (1 - (SSIM+1)/2).sum() + kl_div


class CVAELossFn(torch.nn.Module):
    """
    # TODO: Implement Conditional Variational AutoEncoder Loss Function
    """
    pass


def ParameterSelector(encoder: Encoder, decoder: Decoder):
    """
    Returns the trainable parameters of the Encoder and Decoder.
    """
    return list(encoder.parameters()) + list(decoder.parameters())


class AETrainer:
    """
    Trainer for the AutoEncoder. The trainer trains the Encoder and Decoder on the given
    DataLoader using the given Loss Function and Optimizer.

    The trainer prints for every 10th minibatch the mean loss and similarity as follows:
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss, similarity))

    The trainer also prints for every epoch the mean loss and similarity as follows:
    print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, loss, similarity))

    After every 5 epochs the trainer saves a 3D TSNE plot of logits of the train set as AE_epoch_{}.png.
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: AELossFn|VAELossFn|CVAELossFn, optimizer: torch.optim.Optimizer, gpu: str
    ):
        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda" if gpu == "T" else "cpu")
        self.train()

    def train(self) -> None:
        """
        Trains the AutoEncoder.
        """
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        for epoch in range(EPOCH):
            total_loss = total_similarity = 0
            loss_count = similarity_count = 0

            for minibatch, (noisy, target, _) in enumerate(self.dataloader):
                noisy, target = noisy.to(self.device), target.to(self.device)
                denoised = self.decoder(self.encoder(noisy))
                loss = self.loss_fn(denoised, target)
                total_loss += loss.item()
                loss_count += 1
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if minibatch % 10 == 0:
                    similarity = self.similarity(target, denoised)
                    total_similarity += similarity
                    similarity_count += 1
                    print(f">>>>> Epoch:{epoch}, Minibatch:{minibatch}, Loss:{loss.item()}, Similarity:{similarity}")

            similarity = total_similarity / similarity_count
            loss = total_loss / loss_count
            print(f"----- Epoch:{epoch}, Loss:{loss}, Similarity:{similarity}")

            if epoch % 5 == 0:
                self.tsne_plot(epoch)

    def similarity(self, target: torch.Tensor, output: torch.Tensor) -> float:
        """
        Computes the Structural Similarity Index between the target and output images.
        """
        scores = []
        for i in range(target.shape[0]):
            image1 = (255 * target[i, 0, :, :].squeeze().detach().cpu().numpy()).astype("uint8")
            image2 = (255 * output[i, 0, :, :].squeeze().detach().cpu().numpy()).astype("uint8")
            scores.append(structural_similarity(image1, image2))
        return sum(scores) / len(scores)

    def tsne_plot(self, epoch: int) -> None:
        """
        Saves a 3D TSNE plot of the logits of the training data.
        """
        logits, labels = [], []
        for noisy, _, label in self.dataloader:
            noisy = noisy.to(self.device)
            logits.append(self.encoder(noisy).detach().cpu().view(-1, 128*2*2))
            labels.append(label.flatten())
        logits = torch.cat(logits, dim=0).view(-1, 128*2*2)
        labels = torch.cat(labels, dim=0).flatten()

        logits = TSNE(n_components=3).fit_transform(logits, labels)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(logits[:, 0], logits[:, 1], logits[:, 2], c=labels, alpha=0.75)
        plt.savefig(f"AE_epoch_{epoch}.png")


class VAETrainer:
    """
    Trainer for the Variational AutoEncoder. The trainer trains the Encoder and Decoder on the given
    DataLoader using the given Loss Function and Optimizer.

    The trainer prints for every 10th minibatch the mean loss and similarity as follows:
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss, similarity))

    The trainer also prints for every epoch the mean loss and similarity as follows:
    print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, loss, similarity))

    After every 5 epochs the trainer saves a 3D TSNE plot of logits of the train set as VAE_epoch_{}.png.
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: AELossFn|VAELossFn|CVAELossFn, optimizer: torch.optim.Optimizer, gpu: bool
    ):
        pass

    def train(self) -> None:
        """
        Trains the Variational AutoEncoder.
        """
        pass


class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.

    For each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss, similarity))

    For each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")

    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass


class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """

    def __init__(self, gpu: bool):
        pass

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        pass


class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """

    def __init__(self, gpu: bool):
        pass

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        pass


class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """

    def save_image(digit, save_path):
        pass


def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1:
        raise Exception("Image of shape [1, H, W] required.")

    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0:
        return float("inf")
    else:
        return 20 * torch.log10(255.0/torch.sqrt(mse)).item()


def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1:
        raise Exception("Image of shape [1, H, W] required.")

    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12

    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()