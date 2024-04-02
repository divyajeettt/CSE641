import os
import torch
import random
import torchvision
from EncDec import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity


from sklearn.mixture import GaussianMixture
from tqdm import tqdm

'''
class AlteredMNIST:
    """
    Represents the given modified MNIST dataset. We try to estimate the original
    mapping of the dataset using 10 Gaussian Mixture Models (one for each class).
    Each augmented image is then mapped to its closest clean image within the
    cluster of the GMM that it belongs to. The images are named "Data/X/X_I_L.png":
        - X: {aug=[augmented], clean=[clean]}
        - I: {Index range(0, 60000)}
        - L: {Labels range(10)}
    :attrs:
        - root: The root directory
        - augmented: The list of paths to augmented images
        - augmented_tensors: Mapping of augmented paths to their loaded image tensors
        - mapping: Mapping of augmented paths to their closest clean images
        - transform: The preprocessing transformation pipeline
        - GMMS: The Gaussian Mixture Models for the clean images
        - clean_clusters: The predicted clusters for each clean image
    """

    def __init__(self):
        self.root = os.getcwd()
        self.augmented = [os.path.join(r"Data/aug", image) for image in os.listdir(os.path.join(self.root, r"Data/aug"))]
        self.augmented_tensors = {}

        self.clean = {str(label): [] for label in range(10)}
        for image in os.listdir(os.path.join(self.root, r"Data/clean")):
            label = image[-5]
            image_path = os.path.join(r"Data/clean", image)
            self.clean[label].append(image_path)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor()
        ])

        # self._create_gmms(n_components=15)
        # self._create_mapping()

        import pickle
        # with open("mapping.pkl", "wb") as f1, open("augmented_tensors.pkl", "wb") as f2:
        #     pickle.dump(self.mapping, f1)
        #     pickle.dump(self.augmented_tensors, f2)
        print("Loading Mappings")
        with open("mapping.pkl", "rb") as f1, open("augmented_tensors.pkl", "rb") as f2:
            self.mapping = pickle.load(f1)
            self.augmented_tensors = pickle.load(f2)

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
        aug = self.augmented_tensors[aug_path]
        clean = self.mapping[aug_path]
        return aug, clean, torch.tensor(int(aug_path[-5]))

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Reads the image at the given path and returns the processed tensor.
        """
        return self.transform(torchvision.transforms.functional.to_pil_image(
            torchvision.io.read_image(os.path.join(self.root, path))
        ))

    def _create_gmms(self, n_components: int) -> None:
        """
        Creates the Gaussian Mixture Models for the clean images.
        """
        self.GMMS = {}
        self.clean_clusters = {}

        for label in range(10):
            print("Creating GMM", label)
            label = str(label)
            clean_data = []
            for clean_path in self.clean[label]:
                clean_image = self._load_image(clean_path)
                clean_data.append(clean_image.flatten())
            clean_data = torch.stack(clean_data).view(-1, 28*28).numpy()

            self.GMMS[label] = GaussianMixture(n_components=n_components)
            self.GMMS[label].fit(clean_data)

            predictions = self.GMMS[label].predict(clean_data)
            self.clean_clusters[label] = {i: torch.zeros((0, 28, 28)) for i in range(n_components)}
            for clean_image, prediction in zip(clean_data, predictions):
                clean_tensor = torch.tensor(clean_image.reshape(1, 28, 28))
                self.clean_clusters[label][prediction] = torch.cat((self.clean_clusters[label][prediction], clean_tensor))

    def _create_mapping(self):
        """
        Creates the mapping of augmented images to clean images.
        """
        print("Creating Mapping")
        self.mapping = {}
        for aug_path in tqdm.tqdm(self.augmented):
            aug_image = self._load_image(aug_path)
            self.augmented_tensors[aug_path] = aug_image
            self.mapping[aug_path] = self._get_closest_image(aug_image, aug_path[-5])

    def _get_closest_image(self, aug_image: torch.Tensor, label: str) -> str:
        """
        Returns the closest clean image to the given augmented image.
        """
        aug_image = aug_image.flatten()
        aug_predicted = self.GMMS[label].predict(aug_image.view(1, -1).numpy())[0]
        clean_images = self.clean_clusters[label][aug_predicted]
        distances = torch.norm(clean_images.view(-1, 784) - aug_image, dim=1)
        closest_index = torch.argmin(distances)
        return self.clean_clusters[label][aug_predicted][closest_index].reshape(1, 28, 28)
'''

class AlteredMNIST:
    def __init__(self):
        self.root = os.getcwd()
        self.augmented = [os.path.join(r"Data/aug", image) for image in os.listdir(os.path.join(self.root, r"Data/aug"))]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor()
        ])

        self.clean_tensors = {str(label): torch.zeros((0, 28, 28)) for label in range(10)}
        for clean_path in tqdm(os.listdir(os.path.join(self.root, r"Data/clean"))):
            label = clean_path[-5]
            image_path = os.path.join(r"Data/clean", clean_path)
            image_tensor = self._load_image(image_path)
            self.clean_tensors[label] = torch.cat((self.clean_tensors[label], image_tensor))

        self.augmented_tensors = {str(label): torch.zeros((0, 28, 28)) for label in range(10)}
        self.aug_paths = {str(label): [] for label in range(10)}
        for aug_path in tqdm(self.augmented):
            label = aug_path[-5]
            image_tensor = self._load_image(aug_path)
            self.augmented_tensors[label] = torch.cat((self.augmented_tensors[label], image_tensor))
            self.aug_paths[label].append(os.path.basename(aug_path))

        self.mapping = {}
        for label in range(10):
            self._create_mapping(str(label))

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

    def _create_mapping(self, label):
        clean_tensors = self.clean_tensors[label]
        clean_blurred_1 = torchvision.transforms.functional.gaussian_blur(clean_tensors, 3, sigma=1.5)
        clean_blurred_2 = torchvision.transforms.functional.gaussian_blur(clean_tensors, 5, sigma=2.25)
        clean_features = torch.abs(clean_blurred_1 - clean_blurred_2).flatten(1)

        aug_tensors = self.augmented_tensors[label]
        aug_blurred_1 = torchvision.transforms.functional.gaussian_blur(aug_tensors, 3, sigma=1.5)
        aug_blurred_2 = torchvision.transforms.functional.gaussian_blur(aug_tensors, 5, sigma=2.25)
        aug_features = torch.abs(aug_blurred_1 - aug_blurred_2).flatten(1)

        similarities = torch.matmul(aug_features, clean_features.T)
        closest_index = torch.argmax(similarities, dim=1)
        closest_images = clean_tensors[closest_index]

        for aug_path, aug_tensor, closest_image in zip(self.aug_paths[label], aug_tensors, closest_images):
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
        - mu: The linear layer for the mean of the latent space
        - logvar: The linear layer for the log variance of the latent space
        - flatten: The flattening layer
        - label_embedding: The embedding layer for the labels
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = torch.nn.Sequential(
            EncoderBlock(in_channels=1, out_channels=4, stride=1),
            EncoderBlock(in_channels=4, out_channels=8, stride=3),
            EncoderBlock(in_channels=8, out_channels=16, stride=3),
            EncoderBlock(in_channels=16, out_channels=32, stride=3),
            EncoderBlock(in_channels=32, out_channels=64, stride=1)
        )
        self.fc = torch.nn.Linear(64*2*2, 64*2*2)
        self.mu = torch.nn.Linear(64*2*2, 64)
        self.logvar = torch.nn.Linear(64*2*2, 64)
        self.flatten = torch.nn.Flatten()
        self.label_embedding = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder.
        """
        return self.fc(self.layers(x).view(-1, 64*2*2)).view(-1, 64, 2, 2)


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
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # torch.nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            # torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder Block.
        """
        residual = x
        out = self.block(x)
        if out.shape != residual.shape:
            residual = self.residual_conv(x)
        return torch.nn.functional.relu(out + residual)


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
            DecoderBlock(in_channels=64, out_channels=32, stride=1),
            DecoderBlock(in_channels=32, out_channels=16, stride=3),
            DecoderBlock(in_channels=16, out_channels=8, stride=3),
            DecoderBlock(in_channels=8, out_channels=4, stride=3),
            DecoderBlock(in_channels=4, out_channels=1, stride=1)
        )
        self.fc = torch.nn.Linear(64, 64*2*2)
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(64, 2, 2))

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
    is a combination of the AELossFn (MSE) and KL Divergence.
    """

    def __init__(self):
        super(VAELossFn, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Loss Function.
        """
        AE_loss = super(VAELossFn, self).forward(output, target)
        KL_DIV = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return AE_loss + KL_DIV


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
        loss_fn: AELossFn|VAELossFn|CVAELossFn, optimizer: torch.optim.Optimizer, gpu: str,
        paradigm: str = "AE"
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
                denoised, loss = self.train_batch(noisy, target, labels)
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

            # if epoch % 10 == 0:
            #     self.save_model()
            #     self.tsne_plot(epoch)

    def train_batch(self, noisy: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Processes a single training batch of noisy and target images and
        returns the denoised images and the loss tensor. Accepts labels to
        be consistent with the CVAE.
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
    Trainer for the Variational version of the AutoEncoder. The TSNE plots are
    saved as VAE_epoch_{}.png after every 10th epoch.
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: VAELossFn|CVAELossFn, optimizer: torch.optim.Optimizer, gpu: str,
        paradigm: str = "VAE"
    ):
        super(VAETrainer, self).__init__(dataloader, encoder, decoder, loss_fn, optimizer, gpu, paradigm=paradigm)

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

    def condition(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Conditions the latent space on the label.
        """
        return z + torch.zeros_like(z)

    def train_batch(self, noisy: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Processes a single training batch of noisy and target images and
        returns the denoised images and the loss tensor. Accepts labels for
        to be consistent with the CVAE.
        """
        h = self.encoder(noisy)
        z, mu, logvar = self.bottleneck(h)
        z = self.condition(z, labels)
        denoised = self.decoder(self.decoder.unflatten(self.decoder.fc(z)))
        return denoised, self.loss_fn(denoised, target, mu, logvar)

    def get_emeddings(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Returns the embeddings for the given noisy image batch.
        """
        return self.bottleneck(self.encoder(noisy))[0]


class CVAE_Trainer(VAETrainer):
    """
    Trainer for the Conditional Variational AutoEncoder. The TSNE plots are saved
    as CVAE_epoch_{}.png after every 10th epoch.
    """

    def __init__(
        self, dataloader: torch.utils.data.DataLoader, encoder: Encoder, decoder: Decoder,
        loss_fn: CVAELossFn, optimizer: torch.optim.Optimizer, gpu: str
    ):
        super(CVAE_Trainer, self).__init__(dataloader, encoder, decoder, loss_fn, optimizer, gpu, paradigm="CVAE")

    def condition(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Conditions the latent space on the label.
        """
        return z + self.encoder.label_embedding(label.float())

    def sample(self, label: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Samples the latent space for the given label.
        """
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, 32).to(self.device)
            z = self.condition(z, label)
            samples = self.decoder(self.decoder.unflatten(self.decoder.fc(z)))
        self.encoder.train()
        self.decoder.train()
        return samples


class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
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
        return image.unsqueeze(0).to(self.device)

    def from_path(self, sample: str, original: str, type: str) -> float:
        """
        Compute similarity score of both 'sample' and 'original' and return in float
        """
        sample = self._load_image(sample)
        original = self._load_image(original)
        denoised = self.get_denoised(sample)
        if type == "SSIM":
            return structure_similarity_index(original, denoised)
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
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
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
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """

    def save_image(digit: int, save_path: str) -> None:
        """
        Save the generated image of the given digit at the save_path.
        """
        pass


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
    # Constants
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