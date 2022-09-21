import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from scan_dataloader import CTPET_Dataset
from autoencoder_registration_model import AE_Registrator

if __name__ == "__main__":
    train_dataset = CTPET_Dataset("dest_tryout")

    train_parameters = {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 1,
    }
    training_loader = DataLoader(train_dataset, **train_parameters)

    # Init model
    encoder_size = [16, 8, 8, 16]
    decoder_size = [8, 8, 1]
    model = AE_Registrator(encoder_size, decoder_size)

    for batch in training_loader:
        X = batch["stacked"]
        registred_batch = model(X)
        print(registred_batch.shape)
        break