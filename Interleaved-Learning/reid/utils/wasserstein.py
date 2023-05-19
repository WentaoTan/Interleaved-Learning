import torch
def Wasserstein(mu, sigma, idx1, idx2):
    # p1 = torch.sum(torch.pow((mu[idx1] - mu[idx2]), 2))
    # p2 = torch.sum(torch.pow(torch.pow(sigma[idx1], 1 / 2) - torch.pow(sigma[idx2], 1 / 2), 2))
    # p1 = torch.sum(torch.pow(u1,2))+torch.sum(torch.pow(u2,2))-2*torch.sum(u1*u2)
    # p2 = torch.sum(torch.pow(torch.pow(sigma,1/2) - torch.pow(sigma, 1/2),2))
    p1 = torch.norm(mu[idx1]-mu[idx2],p=2)
    p2 = torch.norm(torch.pow(sigma[idx1],1/2)-torch.pow(sigma[idx2],1/2),p='fro')
    return p1 + p2