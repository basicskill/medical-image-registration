import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from scan_dataloader import CTPET_Dataset
from autoencoder_registration_model import AE_Registrator
from metrics_and_losses import similarity_loss

if __name__ == "__main__":
    train_dataset = CTPET_Dataset("dest_tryout")

    train_parameters = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 1,
    }
    training_loader = DataLoader(train_dataset, **train_parameters)

    # Init model
    encoder_size = [16, 8, 8, 16]
    decoder_size = [8, 8, 1]
    model = AE_Registrator(encoder_size, decoder_size)
    model.train()
    lr = 1e-4
    opt = Adam(model.parameters(), lr=lr)
    num_of_epochs = 10

    loss_arr = []
    for epoch in range(num_of_epochs):
        print(f"Training {epoch+1}/{num_of_epochs} epoch")
        epoch_loss = 0

        for batch in training_loader:
            opt.zero_grad()

            X = batch["stacked"]
            registred_batch = model(X)
            
            pet_loss = similarity_loss(batch["PET"], registred_batch)
            ct_loss = similarity_loss(batch["CT"], registred_batch)
            loss = pet_loss + ct_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        loss_arr.append(epoch_loss)
        torch.save(model.state_dict(), f"tmp_models/{epoch}.pt")

    plt.plot(loss_arr)
    plt.show()