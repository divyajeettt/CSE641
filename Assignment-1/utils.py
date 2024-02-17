import os
import gzip
import torch
import random
import idx2numpy
import numpy as np
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Any, Callable, Iterator


class MNISTDataset:
    """
    Represents the MNIST dataset. If the dataset is not found in the root directory,
    it will be downloaded automatically from the internet. By default, the the training
    set is loaded.
    :params:
        root: the root directory where the dataset will be stored
        train: if True, the training set will be loaded, otherwise the test set will be loaded
        transform: a callable that takes in an image and applies a transformation to it
        target_transform: a callable that takes in a target and applies a transformation to it
        download: if True, the dataset will be downloaded from the internet and stored in the root directory
    :attrs:
        data: the images in the dataset
        targets: the labels in the dataset
    """

    URLs: tuple[str] = (
        r"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        r"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        r"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        r"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    )

    def __init__(self, root: str, train: bool|None = True, transform: Callable|None = None, target_transform: Callable|None = None, download: bool|None = False):
        """
        Initializes the MNIST Dataset.
        """
        self.root = os.path.join(os.getcwd(), os.path.join(root, "MNIST"))
        self.train = train
        if download and not os.path.exists(self.root):
            self._download()
        self.transform = transform
        self.target_transform = target_transform
        self._load()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Returns the image and target at the given index after applying their respective
        transformations (if specified).
        """
        image, target = Image.fromarray(self.data[index]), int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def _download(self) -> None:
        """
        Downloads the dataset from the internet and stores it in the root directory.
        """
        directory = os.path.join(self.root, "raw")
        os.makedirs(directory, exist_ok=True)
        for url in MNISTDataset.URLs:
            filename = os.path.join(directory, os.path.basename(url))
            destination = filename.replace(".gz", "")
            if os.path.exists(destination):
                continue
            urllib.request.urlretrieve(url, filename)
            with gzip.open(filename, "rb") as f_in, open(destination, "wb") as f_out:
                f_out.write(f_in.read())

    def _load(self) -> None:
        """
        Loads the specified dataset from the root directory.
        """
        prefix = "train" if self.train else "t10k"
        images_filename = os.path.join(self.root, "raw", f"{prefix}-images-idx3-ubyte")
        labels_filename = os.path.join(self.root, "raw", f"{prefix}-labels-idx1-ubyte")
        self.data = idx2numpy.convert_from_file(images_filename)
        self.targets = torch.tensor(idx2numpy.convert_from_file(labels_filename), dtype=torch.int64)


class CustomDataLoader:
    """
    Represents a Custom DataLoader (similar to the one in PyTorch) that can be used to iterate
    over a dataset in batches. The DataLoader can be used to iterate over the dataset multiple times
    and can be shuffled if needed.
    :params:
        dataset: the dataset to be loaded
        batch_size: the number of samples in each batch
        shuffle: if True, the dataset will be shuffled before each iteration
    :attrs:
        dataset_size: the number of samples in the dataset
        indices: the ordered indices of the samples in the dataset
    """

    def __init__(self, dataset: MNISTDataset|Dataset, batch_size: int|None = 1, shuffle: bool|None = False):
        """
        Initializes the DataLoader. See help(CustomDataLoader) for more information.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = len(dataset)
        self.indices = torch.arange(self.dataset_size)

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.
        """
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns an iterator over the dataset.
        """
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the next batch of samples in the dataset.
        """
        if self.current_index >= self.dataset_size:
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index+self.batch_size]
        data_batch = torch.stack([self.dataset[i][0] for i in batch_indices])
        target_batch = self.dataset.targets[batch_indices]
        self.current_index += self.batch_size
        return data_batch, target_batch


class ToTensor:
    """
    Converts a PIL image to a PyTorch tensor. By default, the image
    is scaled to the range [0, 1].
    :params:
        scale: the scale factor to apply to the image
        shift: the shift factor to apply to the image
    """

    def __init__(self, scale: float = 1/255, shift: float = 0.0):
        """
        Initializes the ToTensor transformation. See help(ToTensor) for more information.
        """
        self.scale = scale
        self.shift = shift

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Converts the given PIL image to a PyTorch tensor.
        """
        image = torch.from_numpy(np.array(image, dtype=np.float32).reshape(-1, 28*28))
        return self.scale * image + self.shift


class Layer:
    """
    Abstract class to represent a Layer of a Neural Network. A CustomNeuralNetwork
    is a stack of Layers. Each layer must define a forward, backward, and
    update method.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the output of the layer for the given input.
        :param x: the input to the layer
        """
        raise NotImplementedError

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Returns the gradient of the loss with respect to the input of the layer.
        :param grad: the gradient of the loss with respect to the output of the layer
        """
        raise NotImplementedError

    def update(self, lr: float) -> None:
        """
        Updates the parameters of the layer using the given learning rate.
        :param lr: the learning rate of the optimization algorithm
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Represents a (fully-connected) Linear Layer of a Neural Network. The layer
    is defined by a weight matrix and a bias vector. Of course, technically,
    the layer performs an affine transformation. Weights of the layer are
    initialized using Xavier's Method.
    :params:
        input_size: the number of neurons in the input layer
        output_size: the number of neurons in the output layer
    :attrs:
        weight: the weight matrix of the layer
        bias: the bias vector of the layer
        x: the input to the layer
        weight_grad: the gradient of the loss with respect to the weight matrix
        bias_grad: the gradient of the loss with respect to the bias vector
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initializes the Linear Layer. See help(Linear) for more information.
        """
        self.input_size = input_size
        self.output_size = output_size
        scale = 1 / self.input_size**0.5
        self.weight = torch.rand(input_size, output_size) * 2*scale - scale
        self.bias = torch.rand(output_size) * 2*scale - scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.x = x
            return self.x @ self.weight + self.bias

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight_grad = self.x.T @ grad
            self.bias_grad = grad.sum(axis=0)
            return grad @ self.weight.T

    def update(self, lr: float) -> None:
        with torch.no_grad():
            self.weight -= lr * self.weight_grad
            self.bias -= lr * self.bias_grad


class ReLU(Layer):
    """
    Represents a ReLU Layer of a Neural Network. The layer applies the ReLU
    activation function to the input. It does not have any parameters.
    :attrs:
        x: the input to the layer
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.x = x
            return torch.max(x, torch.zeros_like(x))

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return grad * (self.x >= 0).float()

    def update(self, lr: float) -> None:
        pass


class Sigmoid(Layer):
    """
    Represents a Sigmoid Layer of a Neural Network. The layer applies the Sigmoid
    activation function to the input. It does not have any parameters.
    :attrs:
        sigmoid: the output of the layer
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.sigmoid = 1 / (1 + torch.exp(-x))
            return self.sigmoid

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return grad * self.sigmoid * (1 - self.sigmoid)

    def update(self, lr: float) -> None:
        pass


class BatchNorm(Layer):
    """
    Represents a 1-dimensional Batch Normalization Layer of a Neural Network.
    The layer normalizes the input and scales and shifts the normalized input using
    a learned affine transformation. The layer has learnable parameters.
    :params:
        input_size: the number of neurons in the input layer
        affine: if True, the layer will learn an affine transformation
    :attrs:
        scale: the scale parameter of the layer
        shift: the shift parameter of the layer
    """

    def __init__(self, input_size: int, affine: bool|None = True):
        """
        Initializes the BatchNorm Layer. See help(BatchNorm) for more information.
        """
        self.input_size = input_size
        self.affine = affine
        if self.affine:
            scale = 1 / self.input_size**0.5
            self.scale = torch.rand(input_size) * 2*scale - scale
            self.shift = torch.rand(input_size) * 2*scale - scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.x = x
            self.mean = x.mean(axis=0)
            self.std = torch.clamp(x.std(axis=0), 1e-12, 1e+12)
            self.centered = x - self.mean
            return self.centered / self.std

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.mean_grad = grad.mean(axis=0)
            self.std_grad = (grad * self.centered).mean(axis=0) / self.std
            if self.affine:
                self.scale_grad = (grad * self.x).mean(axis=0) / self.std
                self.shift_grad = grad.mean(axis=0)
                return grad * self.scale / self.std - 1/self.input_size * (self.scale_grad + self.std_grad * self.centered.mean(axis=0) / self.std**2)
            return grad/self.std - 1/self.input_size * (self.mean_grad + self.std_grad * self.centered.mean(axis=0) / self.std**2)

    def update(self, lr: float) -> None:
        with torch.no_grad():
            if self.affine:
                self.scale -= lr * self.scale_grad
                self.shift -= lr * self.shift_grad


class CrossEntropyLoss:
    """
    Represents the Cross Entropy Loss function. The loss function is used to
    measure the error in the output of a Neural Network. Loss can be
    calculated by calling the loss object. The loss function expects the
    output to be raw (non-normalized).
    """

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Returns the loss of the network for the given outputs and targets.
        :param outputs: the output of the network
        :param targets: the target of the network
        """
        with torch.no_grad():
            self.outputs = torch.clamp(outputs.softmax(dim=1), 1e-12, 1-1e-12)
            self.targets = targets
            return -torch.log(self.outputs[range(len(targets)), targets]).mean()

    def backward(self) -> torch.Tensor:
        """
        Returns the gradient of the loss with respect to the output of the network.
        This is the step where backpropagation begins.
        """
        with torch.no_grad():
            return self.outputs - torch.eye(self.outputs.shape[1])[self.targets]


class SGD:
    """
    Represents the Stochastic Gradient Descent optimization algorithm. The
    algorithm is used to update the parameters of a Neural Network.
    :params:
        parameters: the parameters of the network
        lr: the learning rate of the algorithm
    """

    def __init__(self, parameters: list[Layer], lr: float):
        """
        Initializes the SGD optimizer. See help(SGD) for more information.
        """
        self.parameters = parameters
        self.lr = lr

    def step(self, grad: torch.Tensor) -> None:
        """
        Updates the parameters of the network using backpropagation beginning
        with the given gradient of the loss.
        :param grad: the gradient of the loss with respect to the output of the network
        """
        with torch.no_grad():
            for i in range(len(self.parameters)-1, -1, -1):
                grad = self.parameters[i].backward(grad)
                self.parameters[i].update(self.lr)


def plot(model: "NeuralNetwork|CustomNeuralNetwork", epochs: int, save: bool|None = False) -> None:
    """
    Plots the accuracy and loss curves of the given model. The model must have
    LOSSES and ACCURACIES attributes.
    """
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), model.LOSSES[0], label="Train")
    plt.plot(range(1, epochs+1), model.LOSSES[1], label="Test")
    plt.plot(range(1, epochs+1), model.LOSSES[2], label="Validation")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), model.ACCURACIES[0], label="Train")
    plt.plot(range(1, epochs+1), model.ACCURACIES[1], label="Test")
    plt.plot(range(1, epochs+1), model.ACCURACIES[2], label="Validation")
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.yticks(torch.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(True)

    if save: plt.savefig("plot.png")
    plt.show()