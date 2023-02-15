from torch.utils.data import random_split, DataLoader
from cough_dataset import cough_dataloader
import pandas as pd
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np


def patient_split(csv,test_csv,train_csv,seed):
    df = pd.read_csv(csv)
    random.seed(seed)

    data = list(set(df['id'].tolist()))
    random.shuffle(data)
    train_patients = data[:round(len(data)*0.8)]
    test_patients = data[round(len(data)*0.8):]

    test_df = pd.DataFrame(
        {'id': [],
         'wav_list': [],
         'path': [],
         'is_cough': [],
        })

    train_df = pd.DataFrame(
        {'id': [],
         'wav_list': [],
         'path': [],
         'is_cough': [],
        })

    for patient in test_patients:
        df1 = df.loc[df['id'] == patient]
        test_df = pd.concat([test_df,df1])
        
    test_df.to_csv(test_csv,index=False)

    for patient in train_patients:
        df1 = df.loc[df['id'] == patient]
        train_df = pd.concat([train_df,df1])
    
    train_df.to_csv(train_csv,index=False)

def stats(csv):
    df = pd.read_csv(csv)
    patients = len(set(df['id'].tolist()))
    wav_files = len(set(df['wav_list'].tolist()))
    chunks = len(set(df['path'].tolist()))
    positive_count = df['is_cough'].tolist().count(1)
    negative_count = df['is_cough'].tolist().count(0)

    print("Num Patients: " + str(patients))
    print("Num Wav Files: " + str(wav_files))
    print("Num Chunks: " + str(chunks))
    print("Num Positives: " + str(positive_count))
    print("Num Negatives: " + str(negative_count))


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def histogram(file_list,title,file_type):
    dur_list = []
    for file in file_list:
        dur_list.append(librosa.get_duration(filename=file))

    print("Average " + file_type + " Length: " + str(round(sum(dur_list)/len(dur_list),2)) + " (s)")
    dur_list_out = reject_outliers(np.array(dur_list)).tolist()

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.add_subplot(111, frameon=False).axis('off')
    fig.tight_layout()

    ax1.hist(dur_list)
    ax1.set_title(title + " (Outliers Included)")

    ax2.hist(dur_list_out)
    ax2.set_title(title + " (No Outliers)")

    plt.xlabel("Duration")
    plt.ylabel("Frequency")



    plt.show()

def main():
    # patient_split('sewanee_labels.csv','sewanee_test.csv','sewanee_train.csv',925)
    # stats('sewanee_test.csv')
    df = pd.read_csv('sewanee_test.csv')
    wav_list = list(set(df['wav_list'].tolist()))
    histogram(wav_list, "Wav Length Histogram","Wav")
    # chunks_list = list(set(df['path'].tolist()))
    # histogram(chunks_list, "Chunks Length Histogram","Chunk")

if __name__ == '__main__':
    main()
