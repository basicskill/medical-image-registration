from math import floor
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn

from scan_dataloader import CTPET_Dataset, get_file_paths
from autoencoder_registration_model import AE_Registrator
from metrics_and_losses import similarity_loss, indexed_loss, batch_rmse, batch_metrics

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

    # Metrics array
    metrics_arr = {
		"Mean": [],
		"STD": [],
		"Average Gradient": [],
		"Entropy": [],
		"RMSE": [] 
	}

    loss_arr = []
    for epoch in range(num_of_epochs):
        print(f"Training {epoch+1}/{num_of_epochs} epoch")
        epoch_loss = 0

        metrics_epoch = {
            "Mean": 0,
            "STD": 0,
            "Average Gradient": 0,
            "Entropy": 0,
            "RMSE": 0
        }


        for batch in training_loader:
            opt.zero_grad()

            X = batch["stacked"]
            registred_batch = model(X).squeeze()

            # ct_loss = indexed_loss(batch["CT"], registred_batch, criterion, 100)
            # pet_loss = indexed_loss(batch["PET"], registred_batch, criterion, 100)

            ct_loss = criterion(batch["CT"], registred_batch)
            pet_loss = criterion(batch["PET"], registred_batch)

            loss = ct_loss + pet_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

            # Metrics
            bmets = batch_metrics(registred_batch.detach())

            for key in bmets:
                metrics_epoch[key] += bmets[key]
            
            avg_rmse = (batch_rmse(registred_batch, batch["CT"]) + batch_rmse(registred_batch, batch["PET"])) / 2
            metrics_epoch["RMSE"] += avg_rmse.item()

        epoch_loss /= len(training_loader)
        loss_arr.append(epoch_loss)
        torch.save(model.state_dict(), f"{models_folder}/{epoch:0{digits}}.pt")
        print(f"\t Loss: {epoch_loss:.0f}")

        for key in metrics_epoch:
            metrics_arr[key].append(metrics_epoch[key])

        # Lower lr on each epoch after 10th
        if epoch > 10:
            scheduler.step()

    metrics_arr["Loss"] = loss_arr

    print(metrics_arr)

    with open('run_metrics.pkl', 'wb') as f:
        pickle.dump(metrics_arr, f)