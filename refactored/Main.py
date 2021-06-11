import torch

episodes = 2000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")