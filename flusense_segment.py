from pydub import AudioSegment
import os 
import pandas as pd

df = pd.read_csv('flusense_metadata.csv')
file_list = []
labels_list = []
for wav in os.listdir('flusense_data_og/'):
    f = os.path.join('flusense_data_og/', wav)
    if os.path.isfile(f):
        audio = AudioSegment.from_file(file = f,
                                  format = "wav")
        df2 = df.loc[df['filename'] == f[17:]]
        i=0
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




df.to_csv('flusense_labels.csv',index=False)








