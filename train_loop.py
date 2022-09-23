from math import floor
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from scan_dataloader import CTPET_Dataset
from autoencoder_registration_model import AE_Registrator
from metrics_and_losses import similarity_loss, indexed_rmse, weighted_mean

if __name__ == "__main__":
    models_folder = "tmp_models"

    for f in os.listdir(models_folder):
        os.remove(os.path.join(models_folder, f))

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
    model.train()
    lr = 1e-3
    opt = Adam(model.parameters(), lr=lr)
    num_of_epochs = 10
    digits = floor(np.log10(num_of_epochs)) + 1

    criterion = nn.MSELoss()

    loss_arr = []
    for epoch in range(num_of_epochs):
        print(f"Training {epoch+1}/{num_of_epochs} epoch")
        epoch_loss = 0

        for batch in training_loader:
            opt.zero_grad()

            X = batch["stacked"]
            registred_batch = model(X).squeeze()

            # ct_loss = similarity_loss(batch["CT"], registred_batch)
            # pet_loss = similarity_loss(batch["PET"], registred_batch)

            # ct_loss = torch.sqrt(criterion(batch["CT"], registred_batch))
            # pet_loss = torch.sqrt(criterion(batch["PET"], registred_batch))

            # ct_loss = torch.sqrt(torch.mean(torch.square(batch["PET"] - registred_batch)))
            # ct_loss = weighted_mean(batch["CT"], registred_batch)

            ct_loss = indexed_rmse(batch["CT"], registred_batch, criterion)

            loss = ct_loss# + pet_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

            # plt.imshow(registred_batch.detach())
            # plt.colorbar()
            # plt.show()

        epoch_loss /= len(training_loader)
        loss_arr.append(epoch_loss)
        torch.save(model.state_dict(), f"{models_folder}/{epoch:0{digits}}.pt")
        print(epoch_loss)

    plt.plot(loss_arr)
    plt.show()