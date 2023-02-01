import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torchmetrics
from torch.utils.data import random_split, DataLoader
from cough_dataset import cough_dataloader
import pandas as pd


class CoughClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=1)


        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)
        x = x.flatten()

        # Final output
        return x

def get_dataloader(path,seed):
    df = pd.read_csv(path)
    myds = cough_dataloader(df)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_test = num_items - num_train
    train_ds, test_ds = random_split(myds, [num_train, num_test],generator=torch.Generator().manual_seed(seed))

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=2000, shuffle=False)

    return train_dl,test_dl



def training(model, train_dl,test_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
    # correct_prediction = 0
    # total_prediction = 0
    # with torch.no_grad():
    #     for data in test_dl:
    #         inputs, labels = data[0],data[1]

    #         # Get predictions
    #         outputs =  torch.sigmoid(model(inputs))
    #         print(outputs)
    #         print(labels)


    #         # Threshold with sigmoid
    #         threshold = torch.tensor([0.5])
    #         prediction = (outputs>threshold).float()*1
    #         auroc = torchmetrics.AUROC(task="binary")
    #         print(roc_auc_score(labels, prediction))
    #         print(auroc(prediction, labels))
    #         # Count of predictions that matched the target label
    #         correct_prediction += (prediction == labels).sum().item()
    #         total_prediction += prediction.shape[0]


    #     acc = correct_prediction/total_prediction
    #     print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    
    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0], data[1]


            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()
            # Get the predicted class with the highest score
            threshold = torch.tensor([0.5])
            prediction = (outputs>threshold).float()*1


            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        # correct_prediction = 0
        # total_prediction = 0
        # with torch.no_grad():
        #     for data in test_dl:
        #         inputs, labels = data[0],data[1]
        #         # Normalize the inputs
        #         # inputs_m, inputs_s = inputs.mean(), inputs.std()
        #         # inputs = (inputs - inputs_m) / inputs_s

        #         # Get predictions
        #         outputs =  torch.sigmoid(model(inputs))


        #         # Threshold with sigmoid
        #         threshold = torch.tensor([0.5])
        #         prediction = (outputs>threshold).float()*1
        #         auroc = torchmetrics.AUROC(task="binary")
        #         print(auroc(prediction, labels))
        #         # Count of predictions that matched the target label
        #         correct_prediction += (prediction == labels).sum().item()
        #         total_prediction += prediction.shape[0]


        #     acc = correct_prediction/total_prediction
        #     print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

    print('Finished Training')

def test(model, test_dl):
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data[0],data[1]
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs =  torch.sigmoid(model(inputs))


            # Threshold with sigmoid
            threshold = torch.tensor([0.5])
            prediction = (outputs>threshold).float()*1
            auroc = torchmetrics.AUROC(task="binary")
            print(auroc(prediction, labels))
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]


    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


            
def accuracy(prediction,labels,threshold):
    prediction = (outputs>torch.tensor([threshold])).float()*1
    correct_prediction += (prediction == labels).sum().item()
    total_prediction += prediction.shape[0]
    return correct_prediction/total_prediction

