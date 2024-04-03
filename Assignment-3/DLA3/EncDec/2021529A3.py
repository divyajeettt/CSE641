import os
import torch
import random
import torchvision
from EncDec import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity


class AlteredMNIST(torch.utils.data.Dataset):
    """
    Represents the given modified MNIST dataset. Due to the unavailability of the
    original mapping for augmentation, we estimate the mapping using the Gaussian
    difference method, which takes the difference of the images blurred with different
    gaussian kernels and finds the closest clean image. The idea is that since
    Gaussian noise has mean 0, the difference of the blurred images should be similar
    to the difference of the clean images. The images are named "Data/X/X_I_L.png":
        - X: {aug=[augmented], clean=[clean]}
        - I: {Index range(0, 60000)}
        - L: {Labels range(10)}
    :attrs:
        - root: The root directory
        - augmented: The list of paths to all augmented images
        - transform: The preprocessing transformation pipeline
        - clean_tensors: The (concatenated) clean tensors for each label
        - augmented_tensors: The (concatenated) augmented tensors for each label
        - augmented_paths: The ordered list of paths of the augmented images for each label
        - mapping: The mapping of augmented image paths to its (augmented_tensor, clean_tensor)
    """

    root: str
    augmented: list[str]
    transform: torchvision.transforms.Compose
    clean_tensors: dict[str, torch.Tensor]
    augmented_tensors: dict[str, torch.Tensor]
    augmented_paths: dict[str, list[str]]
    mapping: dict[str, tuple[torch.Tensor]]

    def __init__(self):
        self.root = os.getcwd()
        self.augmented = [os.path.join(r"Data/aug", image) for image in os.listdir(os.path.join(self.root, r"Data/aug"))]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor()
        ])

        self.clean_tensors = {str(label): torch.zeros((0, 28, 28)) for label in range(10)}
        for clean_path in os.listdir(os.path.join(self.root, r"Data/clean")):
            label = clean_path[-5]
            image_path = os.path.join(r"Data/clean", clean_path)
            image_tensor = self._load_image(image_path)
            self.clean_tensors[label] = torch.cat((self.clean_tensors[label], image_tensor))

        self.augmented_tensors = {str(label): torch.zeros((0, 28, 28)) for label in range(10)}
        self.augmented_paths = {str(label): [] for label in range(10)}
        for aug_path in self.augmented:
            label = aug_path[-5]
            image_tensor = self._load_image(aug_path)
            self.augmented_tensors[label] = torch.cat((self.augmented_tensors[label], image_tensor))
            self.augmented_paths[label].append(os.path.basename(aug_path))

        self.mapping = {}
        for label in range(10):
            self._create_mapping(str(label))

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.augmented)

    def __getitem__(self, index: int) -> None:
        """
        Returns the augmented and clean image pair with the label of the image
        at the given index.
        """
        aug_path = os.path.basename(self.augmented[index])
        aug, clean = self.mapping[aug_path]
        return aug, clean, torch.tensor(int(aug_path[-5]))

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Reads the image at the given path and returns the processed tensor.
        """
        return self.transform(torchvision.transforms.functional.to_pil_image(
            torchvision.io.read_image(os.path.join(self.root, path))
        ))

    def _gaussian_difference(self, label: str, clean: bool = True) -> torch.Tensor:
        """
        Returns the Gaussian difference for the tensors of the given label
        """
        tensors = self.clean_tensors[label] if clean else self.augmented_tensors[label]
        blurred_1 = torchvision.transforms.functional.gaussian_blur(tensors, 3, sigma=0.3)
        blurred_2 = torchvision.transforms.functional.gaussian_blur(tensors, 5, sigma=0.9)
        return torch.abs(blurred_1 - blurred_2).flatten(1)

    def _create_mapping(self, label: str) -> None:
        """
        Creates the mapping of augmented image paths to its (augmented_tensor, clean_tensor)
        by mapping each augmented image to the clean image that maximizes the similarity
        score between the Gaussian differences.
        """
        clean_features = self._gaussian_difference(label, clean=True)
        aug_features = self._gaussian_difference(label, clean=False)

        similarities = torch.matmul(aug_features, clean_features.T)
        aug_mean = torch.norm(aug_features, dim=1, keepdim=True).expand_as(similarities)
        clean_mean = torch.transpose(torch.norm(clean_features, dim=1, keepdim=True), 0, 1).expand_as(similarities)
        closest_index = torch.argmax(similarities/(aug_mean*clean_mean), dim=1)
        closest_images = clean_tensors[closest_index]

        for aug_path, aug_tensor, closest_image in zip(self.augmented_paths[label], aug_tensors, closest_images):
            self.mapping[aug_path] = (aug_tensor.unsqueeze(0), closest_image.unsqueeze(0))


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
        - fc: The layer to capture global features after the Encoder Blocks
        - mu: The linear layer for the mean of the latent space
        - logvar: The linear layer for the log variance of the latent space
        - flatten: The flattening layer
        - label_embedding: The embedding layer for the labels for the CVAE
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
        self.fc = torch.nn.Linear(128*2*2, 128*2*2)
        self.mu = torch.nn.Linear(128*2*2, 128)
        self.logvar = torch.nn.Linear(128*2*2, 128)
        self.flatten = torch.nn.Flatten()
        self.label_embedding = torch.nn.Sequential(
            torch.nn.Linear(10, 128),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder.
        """
        return self.fc(self.layers(x).view(-1, 128*2*2)).view(-1, 128, 2, 2)


class DecoderBlock(torch.nn.Module):
    """
    Represents a Residual Decoder Block for the AutoEncoder. The block is built with
    the ResNet style specifications:
        - Transposed Convolutional Layer with kernel size 3x3
        - ReLU Activation Layer
        - Transposed Convolutional Layer with kernel size 3x3
        - Residual Connection
    :attrs:
        - layers: The layers of the Decoder Block
        - residual_conv: The convolutional layer for the residual connection
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DecoderBlock, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        )
        self.residual_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
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
    Blocks (to approximately undo the encoding). The output images are of shape
    [batch_size, 1, 28, 28].
    :attrs:
        - layers: The layers of the Decoder
        - fc: The linear layer for transforming from the latent space to the original shape
        - unflatten: The unflattening layer
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
        self.fc = torch.nn.Linear(128, 128*2*2)
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(128, 2, 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder.
        """
        return self.layers(z)


class AELossFn(torch.nn.Module):
    """
    Represents the Loss Function for the AutoEncoder. The loss function is the
    MSE between the output and target.
    """

    def __init__(self):
        super(AELossFn, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Loss Function.
        """
        return torch.nn.functional.mse_loss(output, target)


class VAELossFn(AELossFn):
    """
    Represents the Loss Function for the Variational AutoEncoder. The loss function
    is a combination of the AELossFn (MSE) and KL Divergence. The KL Divergence is
    annealed over the epochs with weight 0 <= kl_weight <= 1.
    :attrs:
        - kl_weight: The weight for the KL Divergence
        - anneal_epochs: The number of epochs to anneal the KL Divergence
    """

    def __init__(self):
        super(VAELossFn, self).__init__()
        self.kl_weight = 0.0
        self.anneal_epochs = EPOCH

    def forward(self, output: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Forward pass for the Loss Function.
        """
        AE_loss = super(VAELossFn, self).forward(output, target)
        KL_DIV = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.kl_weight = 1 - 0.5 ** (epoch / (self.anneal_epochs - 1)) if epoch < self.anneal_epochs else 1.0
        return AE_loss + self.kl_weight*KL_DIV


class CVAELossFn(VAELossFn):
    """
    Represents the Loss Function for the Conditional Variational AutoEncoder. The loss
    the same as the VAE Loss Function - a combination of MSE, and KL Divergence.
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
    After every 10 epochs, the trainer saves:
        - a 3D TSNE plot of logits of the train set as AE_epoch_{}.png.
        - the Encoder and Decoder models as AE_encoder.pth and AE_decoder.pth.
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
        loss_fn: AELossFn, optimizer: torch.optim.Optimizer, gpu: str, paradigm: str = "AE"
    ):
        self.paradigm = paradigm
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

        for epoch in range(1, EPOCH+1):
            total_loss = total_similarity = 0
            loss_count = similarity_count = 0

            for minibatch, (noisy, target, labels) in enumerate(self.dataloader):
                noisy, target = noisy.to(self.device), target.to(self.device)
                denoised, loss = self.train_batch(noisy, target, labels, epoch)
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

            if epoch % 10 == 0:
                self.save_model()
                self.tsne_plot(epoch)

    def train_batch(self, noisy: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, epoch: int) -> None:
        """
        Processes a single training batch of noisy and target images and
        returns the denoised images and the loss tensor.
        Accepts epoch to be consistent with the VAE for KL Divergence annealing.
        Accepts labels to be consistent with the CVAE for conditioning.
        """
        z = self.encoder(noisy)
        denoised = self.decoder(z)
        return denoised, self.loss_fn(denoised, target)

    def similarity(self, target: torch.Tensor, output: torch.Tensor) -> float:
        """
        Computes the Structural Similarity Index between the target and output images.
        Uses sklearn.metrics.structural_similarity as the SSIM calculator.
        """
        scores = []
        for i in range(target.shape[0]):
            image1 = (255 * target[i, 0, :, :]).cpu().to(torch.uint8).numpy()
            image2 = (255 * output[i, 0, :, :]).cpu().to(torch.uint8).numpy()
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
                if i % 10 != 0: continue
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

    def save_model(self) -> None:
        """
        Saves the Encoder and Decoder models.
        """
        torch.save(self.encoder.state_dict(), f"{self.paradigm}_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.paradigm}_decoder.pth")


class VAETrainer(AETrainer):
    """
    Trainer for the Variational version of the AutoEncoder. The paradigm is set to VAE.
    After every 10 epochs, the trainer saves:
        - a 3D TSNE plot of logits of the train set as VAE_epoch_{}.png.
        - the Encoder and Decoder models as VAE_encoder.pth and VAE_decoder.pth.
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: VAELossFn, optimizer: torch.optim.Optimizer, gpu: str, paradigm: str = "VAE"
    ):
        super(VAETrainer, self).__init__(
            dataloader, encoder, decoder, loss_fn, optimizer, gpu, paradigm=paradigm
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterizes the latent space to sample from the normal distribution.
        """
        return torch.normal(mu, torch.exp(0.5*logvar))

    def bottleneck(self, h: torch.Tensor) -> None:
        """
        Processes the embeddings to get the latent space and the mean and log variance.
        """
        embeddings = self.encoder.flatten(h)
        mu, logvar = self.encoder.mu(embeddings), self.encoder.logvar(embeddings)
        logits = self.reparameterize(mu, logvar)
        return logits, mu, logvar

    def condition(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Dummy function to be consistent with the CVAE. Simply returns the latent space.
        """
        return z + torch.zeros_like(z)

    def train_batch(self, noisy: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, epoch: int) -> None:
        """
        Processes a single training batch of noisy and target images and
        returns the denoised images and the loss tensor. Accepts labels for
        to be consistent with the CVAE.
        """
        h = self.encoder(noisy)
        z, mu, logvar = self.bottleneck(h)
        z = self.condition(z, labels)
        denoised = self.decoder(self.decoder.unflatten(self.decoder.fc(z)))
        return denoised, self.loss_fn(denoised, target, mu, logvar, epoch)

    def get_emeddings(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Returns the embeddings for the given noisy image batch.
        """
        return self.bottleneck(self.encoder(noisy))[0]


class CVAE_Trainer(VAETrainer):
    """
    Trainer for the Conditional Variational AutoEncoder. The paradigm is set to CVAE.
    After every 10 epochs, the trainer saves:
        - a 3D TSNE plot of logits of the train set as CVAE_epoch_{}.png.
        - the Encoder and Decoder models as CVAE_encoder.pth and CVAE_decoder.pth.
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: CVAELossFn, optimizer: torch.optim.Optimizer, gpu: str = "F"
    ):
        super(CVAE_Trainer, self).__init__(
            dataloader, encoder, decoder, loss_fn, optimizer, gpu, paradigm="CVAE"
        )

    def condition(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Conditions the latent space on the label.
        """
        label = label.to(self.device)
        label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=10)
        return z + self.encoder.label_embedding(label.float())


class AE_TRAINED:
    """
    Loads the trained Encoder-Decoder from saved checkpoints for AE paradigm in
    .eval() mode on the given device and provides methods to compute similarity scores.
    """

    def __init__(self, gpu: bool, paradigm: str = "AE"):
        self.paradigm = paradigm
        self.gpu = gpu
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor()
        ])
        self.device = torch.device("cuda" if gpu else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the Encoder and Decoder models in .eval() mode.
        """
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.encoder.load_state_dict(torch.load(f"{self.paradigm}_encoder.pth"))
        self.decoder.load_state_dict(torch.load(f"{self.paradigm}_decoder.pth"))
        self.encoder.eval()
        self.decoder.eval()

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Reads the image at the given path and returns the processed tensor.
        """
        image = torchvision.io.read_image(path)
        image = torchvision.transforms.functional.to_pil_image(image)
        return self.transform(image).unsqueeze(0).to(self.device)

    def from_path(self, sample: str, original: str, type: str) -> float:
        """
        Computes the similarity score between the denoised image and the original image.
        """
        sample = self._load_image(sample)
        original = self._load_image(original)
        denoised = self.get_denoised(sample.to(self.device))
        if type == "SSIM":
            original = original.numpy().view(28, 28)
            denoised = denoised.numpy().view(28, 28)
            return structural_similarity(original, denoised)
        else:
            return peak_signal_to_noise_ratio(original, denoised)

    def get_denoised(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Returns the denoised image for the given noisy image.
        Input Shape: [1, 1, H, W]
        Output Shape: [1, H, W]
        """
        with torch.no_grad():
            z = self.encoder(noisy)
            denoised = self.decoder(z)
        return denoised.squeeze(0)


class VAE_TRAINED(AE_TRAINED):
    """
    Loads the trained Encoder-Decoder from saved checkpoints for VAE paradigm in
    .eval() mode on the given device and provides methods to compute similarity scores.
    """

    def __init__(self, gpu: bool):
        super(VAE_TRAINED, self).__init__(gpu, paradigm="VAE")

    def get_denoised(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Returns the denoised image for the given noisy image.
        Input Shape: [1, 1, H, W]
        Output Shape: [1, H, W]
        """
        with torch.no_grad():
            h = self.encoder(noisy)
            z, _, _ = self.bottleneck(h)
            denoised = self.decoder(self.decoder.unflatten(self.decoder.fc(z)))
        return denoised.squeeze(0)


class CVAE_Generator:
    """
    Generator for the Conditional Variational AutoEncoder. The generator is used to
    generate images of a given digit. The generator loads the trained Encoder and
    Decoder models in .eval() mode and provides a method to save the generated image.
    """

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.load_state_dict(torch.load("CVAE_encoder.pth"))
        self.decoder.load_state_dict(torch.load("CVAE_decoder.pth"))
        self.encoder.eval()
        self.decoder.eval()

    def save_image(digit: int, save_path: str) -> None:
        """
        Generates and save the generated image of the given digit at the save_path.
        """
        with torch.no_grad():
            z = torch.randn(1, 128)
            label = torch.nn.functional.one_hot(torch.tensor([digit]).to(torch.int64), num_classes=10)
            z = z + self.encoder.label_embedding(label.float())
            image = self.decoder(self.decoder.unflatten(self.decoder.fc(z)))
            torchvision.io.write_image(image, save_path)


def peak_signal_to_noise_ratio(img1: torch.Tensor, img2: torch.Tensor) -> float:
    if img1.shape[0] != 1:
        raise Exception("Image of shape [1, H, W] required.")

    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0:
        return float("inf")
    else:
        return 20 * torch.log10(255.0/torch.sqrt(mse)).item()


def structure_similarity_index(img1: torch.Tensor, img2: torch.Tensor) -> float:
    if img1.shape[0] != 1:
        raise Exception("Image of shape [1, H, W] required.")

    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = torch.nn.functional.conv2d(img1*img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = torch.nn.functional.conv2d(img2*img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  torch.nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=channels) - mu12

    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()