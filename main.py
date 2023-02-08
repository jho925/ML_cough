from model import CoughClassifier, training, test, get_dataloader
import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision
from torchvision  import datasets, models, transforms


def main():
    resnet_model = models.resnet34(pretrained=True)
    nr_filters = resnet_model.fc.in_features  #number of input features of last layer
    resnet_model.fc = nn.Linear(nr_filters, 1)
    resnet_model.conv1 = nn.Conv2d(2, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    train_dl, test_dl = get_dataloader('ML_labels.csv',0)
    training(resnet_model, train_dl,test_dl, 30)
    torch.save(resnet_model.state_dict(), 'model.pth')
    test(resnet_model, test_dl)
    # roc_auc(Model,test_dl)

if __name__ == '__main__':
    main()