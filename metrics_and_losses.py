import torch
import numpy as np

def similarity_loss(original_img: torch.Tensor, new_img: torch.Tensor):

	# print(original_img.shape)
	# print(new_img.shape)

	# Define loss constants
	C1 = .01
	C2 = .03

	# Calculate means
	m_orig = torch.mean(original_img, dim=[1, 2], keepdim=True)
	m_new = torch.mean(new_img, dim=[1, 2], keepdim=True)

	# Calculate standard deviation
	var_orig = torch.var(original_img, dim=[1, 2], keepdim=True)
	var_new = torch.var(new_img, dim=[1, 2], keepdim=True)

	# Calculate covariance
	covar = torch.sum((original_img - m_orig) * (new_img - m_new) / (len(original_img - 1)), dim=[1, 2])

	ssim = ((2 * m_orig * m_new + C1) / (m_orig**2 + m_new**2 + C1)) * \
			((2 * covar + C2) / (var_orig + var_new + C2))
	ssim = torch.mean(ssim)

	return -ssim

def indexed_loss(original_img: torch.Tensor, new_img: torch.Tensor, criterion, thr: float):

	indexes = original_img > thr

	return criterion(new_img[indexes], original_img[indexes])


def weighted_mean(orig, new):
	return torch.mean(torch.square(50 * (orig - new)))


def batch_metrics(batch_fused: torch.Tensor) -> dict:
	# Mean
	mean = torch.mean(batch_fused).item()

	# STD
	std = torch.std(batch_fused, dim=[1, 2])
	std = torch.mean(std).item()

	# Average gradient
	Gx = batch_fused[:, :, 0:-1] - batch_fused[:, :, 1:]
	Gy = batch_fused[:, 0:-1, :] - batch_fused[:, 1:, :]
	Gx = Gx[:, :-1, :]
	Gy = Gy[:, :, :-1]

	ag = torch.mean(torch.sqrt((Gx**2 + Gy**2) / 2), dim=[1, 2])
	ag = torch.mean(ag).item()

	# Entropy
	ent = batch_entropy(batch_fused)

	return {
		"Mean": mean,
		"STD": std,
		"Average Gradient": ag,
		"Entropy": ent,
	}


def batch_entropy(img_batch):
	ent = 0
	for idx in range(img_batch.shape[0]):
		cnt, _ = np.histogram(img_batch[0, :, :].flatten(), bins=50)
		prob = (cnt + 1) / np.max(cnt)
		ent += -np.sum(prob * np.log(prob))
	
	ent /= img_batch.shape[0]

	return ent

def batch_rmse(img1, img2):
	rmse = torch.sqrt(torch.mean((img1 - img2)**2, dim=[1, 2]))
	rmse = torch.mean(rmse)

	return rmse