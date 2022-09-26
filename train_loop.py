from math import floor
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn

from scan_dataloader import CTPET_Dataset, get_file_paths
from autoencoder_registration_model import AE_Registrator
from metrics_and_losses import similarity_loss, indexed_loss, weighted_mean

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


if __name__ == "__main__":
    models_folder = "tmp_models"

    for f in os.listdir(models_folder):
        os.remove(os.path.join(models_folder, f))

    ct_pths, pet_pths = get_file_paths("dest_tryout")
    train_dataset = CTPET_Dataset(ct_pths, pet_pths)

    train_parameters = {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 1,
    }
    training_loader = DataLoader(train_dataset, **train_parameters)

    # Init model
    # encoder_size = [64, 32, 32, 64]
    # decoder_size = [32, 32, 1]
    encoder_size = [32, 32, 16, 64]
    decoder_size = [32, 32, 1]
    model = AE_Registrator(encoder_size, decoder_size)
    model.train()
    lr = 1e-3
    opt = Adam(model.parameters(), lr=lr, weight_decay=2e-5)
    num_of_epochs = 30
    digits = floor(np.log10(num_of_epochs)) + 1
    scheduler = ExponentialLR(opt, gamma=0.9)

    # criterion = nn.MSELoss()
    criterion = similarity_loss

    # Use cuda if available
    # model.load_state_dict(torch.load("./tmp_models/11.pt"))
    model.to(device)

    loss_arr = []
    for epoch in range(num_of_epochs):
        print(f"Training {epoch+1}/{num_of_epochs} epoch")
        epoch_loss = 0

        for batch in training_loader:
            opt.zero_grad()

            X = batch["stacked"]
            registred_batch = model(X).squeeze()

            # ct_loss = indexed_loss(batch["CT"], registred_batch, criterion, 100)
            # pet_loss = indexed_loss(batch["PET"], registred_batch, criterion, 100)

            ct_loss = criterion(batch["CT"], registred_batch)
            pet_loss = criterion(batch["PET"], registred_batch)


            if epoch > 20:
                a = 1
                b = 1
            else:
                a = 1
                b = 1

            loss = a * ct_loss + b * pet_loss
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