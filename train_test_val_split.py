from torch.utils.data import random_split, DataLoader
from cough_dataset import cough_dataloader
import pandas as pd

df = pd.read_csv('sewanee_labels.csv')
myds = cough_dataloader(df)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_test = num_items - num_train
train_ds, test_ds = random_split(myds, [num_train, num_test])

# Create training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
