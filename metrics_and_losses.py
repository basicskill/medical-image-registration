import torch


def similarity_loss(original_img: torch.Tensor, new_img: torch.Tensor):

    # Define loss constants
    C1 = .01
    C2 = .03

    # Calculate means
    m_orig = torch.mean(original_img)
    m_new = torch.mean(new_img)

    # Calculate standard deviation
    var_orig = torch.var(original_img)
    var_new = torch.var(new_img)

    # Calculate covariance
    covar = torch.sum((original_img - m_orig) * (new_img - m_new) / (len(original_img - 1)))

    ssim = ((2 * m_orig * m_new + C1) / (m_orig**2 + m_new**2 + C1)) * \
            ((2 * covar + C2) / (var_orig + var_new + C2))

    return 1 - ssim
