from pydub import AudioSegment
import os 
import pandas as pd

df = pd.read_csv('flusense_metadata.csv')
file_list = []
labels_list = []
for wav in os.listdir('flusense_data_og/'):
    f = os.path.join('flusense_data_og/', wav)
    if f.endswith(".wav"):
        audio = AudioSegment.from_file(file = f,
                                  format = "wav")
        df2 = df.loc[df['filename'] == f[17:]]
        i=0
        if df2.shape[0] == 1:
            new_filename = 'flusense_data/' + f[17:]
            file_list.append(new_filename)
            if row['label'] == 'cough':
                labels_list.append(1)
            else:
                labels_list.append(0)
            curr_chunk = audio
            curr_chunk.export('flusense_data/' + f[17:], format="wav")
        else:
            for index, row in df2.iterrows():
                i+=1
                new_filename = 'flusense_data/' + f[17:-4] + 'segment' + str(i) + '.wav'
                file_list.append(new_filename)
                if row['label'] == 'cough':
                    labels_list.append(1)
                else:
                    labels_list.append(0)
                curr_chunk = audio[row['start']*1000:row['end']*1000]
                curr_chunk.export('flusense_data/' + f[17:-4] + 'segment' + str(i) + '.wav', format="wav")

df = pd.DataFrame(
    {'path': file_list,
     'is_cough': labels_list,
    })

df2 = pd.read_csv('sewanee_coughs_dataset.csv')
file_list = []
labels_list = []
for wav in os.listdir('sewanee_sounds/'):
    f = os.path.join('sewanee_sounds/', wav)
    if f.endswith(".wav"):
        audio = AudioSegment.from_file(file = f,
                                  format = "wav")
        f1 = lambda x: x['wav_file'].split('/')[-1]
        df2['wav_file'] = df2.apply(f1, axis=1)
        df3 = df2.loc[df2['wav_file'] == f[15:]]
        if df3.empty:
            continue
        
        if df3.shape[0] == 1:
            new_filename = 'sewanee_data/' + f[15:]
            file_list.append(new_filename)
            if df3['is_cough'].iloc[0] == True:
                labels_list.append(1)
            else:
                labels_list.append(0)
            curr_chunk = audio
            curr_chunk.export('sewanee_data/' + f[15:], format="wav")
        else:
            i=0
            time_list = df3['timestamp_ms'].values.tolist()
            time_list = [x - time_list[0] for x in time_list]
            for index, row in df3.iterrows():
                i+=1
                new_filename = 'sewanee_data/' + f[15:-4] + 'segment' + str(i) + '.wav'
                file_list.append(new_filename)
                if row['is_cough'] == 'TRUE':
                    labels_list.append(1)
                else:
                    labels_list.append(0)

                if i == len(time_list):
                    curr_chunk = audio[time_list[i-1]:]
                else:
                    curr_chunk = audio[time_list[i-1]:time_list[i]]
                curr_chunk.export('sewanee_data/' + f[15:-4] + 'segment' + str(i) + '.wav', format="wav")

df2 = pd.DataFrame(
    {'path': file_list,
     'is_cough': labels_list,
    })

df3 = pd.concat([df, df2], axis=0)


df3.to_csv('ML_labels.csv',index=False)








