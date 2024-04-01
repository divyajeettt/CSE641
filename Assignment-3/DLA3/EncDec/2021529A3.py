import os
import torch
import random
import torchvision
from EncDec import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity

'''
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
        aug = self.get_pil_image(aug_path)
        clean = self.get_pil_image(self.mapping[aug_path])
        return self.transform(aug), self.transform(clean), torch.tensor(int(aug_path[-5]))

    def get_pil_image(self, path: str) -> "PIL.Image.Image":
        """
        Reads the image at the given path and returns the PIL image.
        """
        return torchvision.transforms.functional.to_pil_image(
            torchvision.io.read_image(os.path.join(self.root, path))
        )
'''

from sklearn.mixture import GaussianMixture
import tqdm


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

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor()
        ])

        self._create_gmms()
        self._create_mapping()

    def __len__(self) -> int:
        return len(self.augmented)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        aug_path = self.augmented[index]
        aug = self._get_pil_image(aug_path)
        clean = self._get_pil_image(self.mapping[aug_path])
        return self.transform(aug), self.transform(clean), torch.tensor(int(aug_path[-5]))

    def _get_pil_image(self, path: str) -> "PIL.Image.Image":
        return torchvision.transforms.functional.to_pil_image(
            torchvision.io.read_image(os.path.join(self.root, path))
        )

    def _get_feature_vector(self, path: str) -> torch.Tensor:
        return self.transform(self._get_pil_image(path)).flatten()

    def _create_gmms(self):
        self.GMMS = {}
        for label in range(10):
            print("Creating GMM", label)
            self.GMMS[str(label)] = GaussianMixture(n_components=10)
            clean_images = []
            for clean_path in self.clean[str(label)]:
                clean_images.append(self._get_feature_vector(clean_path))

            clean_images = torch.stack(clean_images).view(-1, 28*28).numpy()
            self.GMMS[str(label)].fit(clean_images)

    def _create_mapping(self):
        print("Creating Mapping")
        self.mapping = {}
        for aug_path in tqdm.tqdm(self.augmented):
            self.mapping[aug_path] = self._get_closest_image(aug_path)

    def _get_closest_image(self, aug_path: str) -> str:
        aug = self._get_feature_vector(aug_path)
        label = aug_path[-5]
        likelihoods = self.GMMS[label].score_samples(aug.reshape(1, -1))
        closest = torch.argmax(torch.tensor(likelihoods)).item()
        return self.clean[label][closest]


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


'''
class EncoderBlockDownsample(torch.nn.Module):
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
        self.downsample = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        residual = [self.downsample(x)] * (out.shape[1] // x.shape[1])
        residual = torch.concatenate(residual, dim=1)
        return torch.nn.functional.relu(out + residual[:, :, :out.shape[2], :out.shape[3]])
'''


class Encoder(torch.nn.Module):
    """
    Represents the Encoder for the AutoEncoder. The Encoder consists of 5 Encoder
    Blocks. The encoded logits/embeddings are of shape [batch_size, 128, 2, 2].
    :attrs:
        - layers: The layers of the Encoder
        - mu: The linear layer for the mean of the latent space
        - logvar: The linear layer for the log variance of the latent space
        - flatten: The flattening layer
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = torch.nn.Sequential(
            EncoderBlock(in_channels=1, out_channels=4, stride=1),
            EncoderBlock(in_channels=4, out_channels=8, stride=3),
            EncoderBlock(in_channels=8, out_channels=16, stride=3),
            EncoderBlock(in_channels=16, out_channels=32, stride=3)
        )
        self.mu = torch.nn.Linear(32*2*2, 32)
        self.logvar = torch.nn.Linear(32*2*2, 32)
        self.flatten = torch.nn.Flatten()

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


'''
class DecoderBlockUpsample(torch.nn.Module):
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
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.upsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        residual = torch.nn.functional.interpolate(self.upsample(x), size=out.shape[2:], mode="bicubic")
        return torch.nn.functional.relu(out + residual)
'''


class Decoder(torch.nn.Module):
    """
    Represents the Decoder for the AutoEncoder. The Decoder consists of 5 Decoder
    Blocks (to approximately undo the encoding). The output images are of shape [batch_size, 1, 28, 28].
    :attrs:
        - layers: The layers of the Decoder
        - fc: The linear layer for transforming from the latent space to the initial shape
        - unflatten: The unflattening layer
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = torch.nn.Sequential(
            DecoderBlock(in_channels=32, out_channels=16, stride=3),
            DecoderBlock(in_channels=16, out_channels=8, stride=3),
            DecoderBlock(in_channels=8, out_channels=4, stride=3),
            DecoderBlock(in_channels=4, out_channels=1, stride=1)
        )
        self.fc = torch.nn.Linear(32, 32*2*2)
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(32, 2, 2))

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
    Trainer for the AutoEncoder. The trainer trains the Encoder and Decoder on
    the given DataLoader using the given Loss Function and Optimizer. The trainer
    prints the mean loss and similarity for every 10th minibatch and for every epoch.
    After every 10 epochs the trainer saves a 3D TSNE plot of logits of the train
    set as AE_epoch_{}.png.
    :attrs:
        - paradigm: The paradigm of the trainer (AE)
        - dataloader: The DataLoader for the training data
        - encoder: The Encoder for the AutoEncoder
        - decoder: The Decoder for the AutoEncoder
        - loss_fn: The Loss Function for the AutoEncoder
        - optimizer: The Optimizer for the AutoEncoder
        - device: The device to train the AutoEncoder on
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: AELossFn|VAELossFn|CVAELossFn, optimizer: torch.optim.Optimizer, gpu: str
    ):
        self.paradigm = "AE"
        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda" if gpu == "T" else "cpu")
        if isinstance(loss_fn, AELossFn): self.train()

    def train(self) -> None:
        """
        Trains the AutoEncoder.
        """
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        for epoch in range(1, EPOCH+1):
            total_loss = total_similarity = 0
            loss_count = similarity_count = 0

            for minibatch, (noisy, target, _) in enumerate(self.dataloader):
                noisy, target = noisy.to(self.device), target.to(self.device)
                denoised, loss = self.train_batch(noisy, target)
                total_loss += loss.item()
                loss_count += 1
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if minibatch % 10 == 0:
                    total_similarity += abs(self.similarity(target, denoised))
                    similarity_count += 1
                    avg_loss = total_loss / loss_count
                    avg_similarity = total_similarity / similarity_count
                    print(f">>>>> Epoch:{epoch}, Minibatch:{minibatch}, Loss:{avg_loss}, Similarity:{avg_similarity}")

            avg_similarity = total_similarity / similarity_count
            avg_loss = total_loss / loss_count
            print(f"----- Epoch:{epoch}, Loss:{avg_loss}, Similarity:{avg_similarity}")

            if epoch % 10 == 0: self.tsne_plot(epoch)

    def train_batch(self, noisy: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Processes a single training batch of noisy and target images and
        returns the denoised images and the loss tensor.
        """
        z = self.encoder(noisy)
        denoised = self.decoder(z)
        return denoised, self.loss_fn(denoised, target)

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

    def get_embeddings(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Returns the embeddings for the given noisy image batch.
        """
        return self.encoder(noisy)

    def tsne_plot(self, epoch: int) -> None:
        """
        Saves a 3D TSNE plot of the logits of the training data.
        """
        self.encoder.eval()
        logits, labels = [], []
        with torch.no_grad():
            for i, (noisy, _, label) in enumerate(self.dataloader):
                if i % 2: continue
                noisy = noisy.to(self.device)
                embeddings = self.get_embeddings(noisy)
                size = embeddings.shape[1] * embeddings.shape[2] * embeddings.shape[3]
                logits.append(embeddings.detach().cpu().view(-1, size))
                labels.append(label.flatten())
            logits = torch.cat(logits, dim=0).view(-1, size)
            labels = torch.cat(labels, dim=0).flatten()

        logits = TSNE(n_components=3).fit_transform(logits, labels)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(logits[:, 0], logits[:, 1], logits[:, 2], c=labels, alpha=0.75)
        plt.savefig(f"{self.paradigm}_epoch_{epoch}.png")
        plt.close()
        self.encoder.train()


class VAETrainer(AETrainer):
    """
    Trainer for the Variational version of the AutoEncoder. The TSNE plots are
    saved as VAE_epoch_{}.png after every 10th epoch.
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: VAELossFn|CVAELossFn, optimizer: torch.optim.Optimizer, gpu: bool
    ):
        super(VAETrainer, self).__init__(dataloader, encoder, decoder, loss_fn, optimizer, gpu)
        self.paradigm = "VAE"
        if isinstance(loss_fn, VAELossFn): self.train()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterizes the latent space to sample from the normal distribution.
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return mu + eps*std

    def bottleneck(self, h: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Processes the embeddings to get the latent space and the mean and log variance.
        """
        embeddings = self.encoder.flatten(h)
        mu, logvar = self.encoder.mu(embeddings), self.encoder.logvar(embeddings)
        logits = self.reparameterize(mu, logvar)
        return logits, mu, logvar

    def train_batch(self, noisy: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Processes a single training batch of noisy and target images and
        returns the denoised images and the loss tensor.
        """
        h = self.encoder(noisy)
        z, mu, logvar = self.bottleneck(h)
        denoised = self.decoder(self.decoder.unflatten(self.decoder.fc(z)))
        return denoised, self.loss_fn(denoised, target, mu, logvar)

    def get_emeddings(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Returns the embeddings for the given noisy image batch.
        """
        return self.bottleneck(self.encoder(noisy))[0]


class CVAE_Trainer(VAETrainer):
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