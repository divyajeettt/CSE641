
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
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
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
    Blocks. The encoded logits/embeddings are of shape [batch_size, 64, 2, 2].
    :attrs:
        - layers: The layers of the Encoder
        - fc: The layer to capture global features
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
        - ReLU Activation Layer
        - Transposed Convolutional Layer with kernel size 3x3
        - Residual Connection
    :attrs:
        - layers: The layers of the Decoder Block
        - residual_conv: The convolutional layer for the residual connection
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DecoderBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.residual_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

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

