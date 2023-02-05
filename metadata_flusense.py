import pandas as pd
from pathlib import Path
import librosa
import os
import moviepy.editor as moviepy


def main():
	# Read metadata file
	df = pd.read_csv('flusense_metadata.csv')

	df2 = pd.DataFrame()

	df2['path'] = 'flusense_data/' + df.loc[df['label'] == 'cough']['filename']

	df2['is_cough'] = 1

	df3 = pd.DataFrame()

	df3['path'] = 'flusense_data/' + df.loc[df['label'] != 'cough']['filename']

	df3['is_cough'] = 0

	df = pd.concat([df2, df3], axis=0)

	df.drop_duplicates(inplace=True)

	df.to_csv('flusense_labels.csv',index=False)




if __name__ == '__main__':
	main()
