import torch

def choose_device():
    """
    选择device，cuda > mps(Metal) > cpu
    :return: device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device