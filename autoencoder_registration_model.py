from turtle import forward
import numpy as np
import torch
import torch.nn as nn


class AE_Registrator(nn.Module):
    """Autoencoder convolutional model for PET and CT registration."""

    def __init__(self, encoder_channels: list, decoder_channels: list) -> None:
        super().__init__()

        # Check if encoder has enough layers
        assert len(encoder_channels) > 1

        encoder_layer_list = [
            nn.Conv2d(2, encoder_channels[0], kernel_size=3, padding=1)
        ]

        # Init all encoder layers
        for idx in range(1, len(encoder_channels)):

            # Add max pool layer
            encoder_layer_list.append(
                nn.MaxPool2d(2)
            )

            # Add convolutional layer
            encoder_layer_list.append(
                nn.Conv2d(encoder_channels[idx - 1], encoder_channels[idx],
                    kernel_size=3, padding=1)
            )
        
        # Create torch encoder object
        self.encoder = nn.ModuleList(encoder_layer_list)

        # Create decoder
        decoder_layer_list = [
            nn.ConvTranspose2d(encoder_channels[-1], decoder_channels[0],
                kernel_size=2, stride=2),
            nn.Conv2d(decoder_channels[0], decoder_channels[1], kernel_size=3, padding=1)
        ]

        # Init decoder layers
        for idx in range(1, len(decoder_channels)):
            
            # Add deconvolution layer
            decoder_layer_list.append(
                nn.ConvTranspose2d(decoder_channels[idx - 1], decoder_channels[idx - 1],
                    kernel_size=2, stride=2)
            )

            # Add convolutional layer
            decoder_layer_list.append(
                nn.Conv2d(decoder_channels[idx - 1], decoder_channels[idx],
                    kernel_size=3, padding=1)
            )
        
        self.decoder = nn.ModuleList(decoder_layer_list)

    def forward(self, X) -> torch.Tensor:

        # Encode batch
        for layer in self.encoder:
            X = layer(X)
            # print(f"{layer} -> {X.shape}")
        
        # Decode batch
        for layer in self.decoder:
            X = layer(X)
            # print(f"{layer} -> {X.shape}")

        return X