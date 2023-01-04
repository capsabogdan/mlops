import torch
import numpy as np
import pdb
from glob import glob

# put the following statement where you want the code to stop and step to pdb
# pdb.set_trace()

def mnist():
    # exchange with the corrupted mnist dataset
    from glob import glob
    train = []
    for file in glob("./corruptmnist/train*"):
        crt = np.load(file)
        train.append(crt)
    print(train)
    test = np.load("./corruptmnist/test.npz")
    #train = torch.randn(50000, 784)
    #test = torch.ra
    # ndn(10000, 784) 
    return train, test

mnist()
