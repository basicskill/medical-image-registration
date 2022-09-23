import torch


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

	return 1 - ssim

def indexed_rmse(original_img: torch.Tensor, new_img: torch.Tensor, criterion):

	indexes = original_img > 100

	# rmse = torch.sqrt(torch.mean(torch.square(original_img[indexes] - new_img[indexes])))

	return criterion(new_img[indexes], original_img[indexes])


def weighted_mean(orig, new):
	return torch.mean(torch.square(50 * (orig - new)))