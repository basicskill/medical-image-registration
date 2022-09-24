from math import floor
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
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
    encoder_size = [32, 16, 16, 32]
    decoder_size = [16, 16, 1]
    model = AE_Registrator(encoder_size, decoder_size)
    model.train()
    lr = 1e-3
    opt = Adam(model.parameters(), lr=lr, weight_decay=2e-5)
    num_of_epochs = 30
    digits = floor(np.log10(num_of_epochs)) + 1
    scheduler = ExponentialLR(opt, gamma=0.9)

    criterion = nn.MSELoss()

    loss_arr = []
    for epoch in range(num_of_epochs):
        print(f"Training {epoch+1}/{num_of_epochs} epoch")
        epoch_loss = 0

        for batch in training_loader:
            opt.zero_grad()

            X = batch["stacked"]
            registred_batch = model(X).squeeze()

            ct_loss = indexed_rmse(batch["CT"], registred_batch, criterion, 100)
            pet_loss = indexed_rmse(batch["PET"], registred_batch, criterion, 100)

            loss = ct_loss + pet_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

            # plt.imshow(registred_batch.detach())
            # plt.colorbar()
            # plt.show()

        epoch_loss /= len(training_loader)
        loss_arr.append(epoch_loss)
        torch.save(model.state_dict(), f"{models_folder}/{epoch:0{digits}}.pt")
        print(f"\t Loss: {epoch_loss:.0f}")

        # Lower lr on each epoch after 10th
        if epoch > 10:
            scheduler.step()

    plt.plot(loss_arr)
    plt.show()