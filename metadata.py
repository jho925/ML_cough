import pandas as pd
from pathlib import Path
import librosa
import os

def main():
	# Read metadata file
	df = pd.read_csv('sewanee_coughs_dataset.csv')
	df.head()

	# Construct file path by concatenating fold and file name

	f = lambda x: x['wav_file'].split('/')[-1]

	df['wav_file'] = df.apply(f, axis=1)


	df = df[['wav_file', 'is_cough']]

	files = df['wav_file'].tolist()

	for f in files:
		if f not in os.listdir('sewanee_sounds/'):
			df = df[df['wav_file'] != f] 

	df['path'] = 'sewanee_sounds/' + df['wav_file']

	max_dur = 0
	for i in df['path']:
		a = librosa.get_duration(filename=i)
		if a> max_dur:
			max_dur = a

	print(a)

	# Take relevant columns
	df = df[['path', 'is_cough']]

	df.to_csv('sewanee_labels.csv',index=False)


if __name__ == '__main__':
	main()