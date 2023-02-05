from model import CoughClassifier, training, test
import torch
from torch.utils.data import random_split, DataLoader
from cough_dataset import cough_dataloader
import pandas as pd
from train_test_val_split import test_dl, train_dl

def main():
    Model = CoughClassifier()
    Model.load_state_dict(torch.load('model.pth'))
    df = pd.read_csv('irl_test.csv')
    myds = cough_dataloader(df)
    cough_dl = DataLoader(myds, batch_size=3, shuffle=False)
    correct_prediction = 0
    total_prediction = 0
    with torch.no_grad():
        for data in cough_dl:
            inputs, labels = data[0],data[1]
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = torch.sigmoid(Model(inputs))
            print(outputs)

            # Threshold with sigmoid
            threshold = torch.tensor([0.5])
            prediction = (outputs>threshold).float()*1
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]


    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')



if __name__ == '__main__':
    main()