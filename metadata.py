import pandas as pd
from pathlib import Path
import librosa
import os

def main():
	# Read metadata file
	df = pd.read_csv('sewanee_coughs_dataset.csv')

	# Construct file path by concatenating fold and file name
	f = lambda x: x['wav_file'].split('/')[-1]

	df['wav_file'] = df.apply(f, axis=1)

	files = df['wav_file'].tolist()

	for f in files:
		if f not in os.listdir('sewanee_sounds/'):
			df = df[df['wav_file'] != f] 

	df1 = pd.DataFrame()

	df1['path'] = 'sewanee_sounds/' + df['wav_file']

	df1.drop_duplicates(inplace=True)

	df1['is_cough'] = 1

	df = pd.read_csv('development_chunks_raw.csv')

	df2 = pd.DataFrame()

	df2['path'] = 'chunks/' + df['file_name'] + '.48kHz.wav'

	df2.drop_duplicates(inplace=True)

	df2['is_cough'] = 0

	df = pd.concat([df1, df2], axis=0)


	max_dur = 0
	for i in df['path']:
		a = librosa.get_duration(filename=i)
		if a> max_dur:
			max_dur = a

	print(a)

	# Take relevant column

	df.to_csv('ML_labels.csv',index=False)

if __name__ == '__main__':
	main()
