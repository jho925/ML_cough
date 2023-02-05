import pandas as pd
from pathlib import Path
import librosa
import os
import moviepy.editor as moviepy
import subprocess


def convert(filename,f_out):
    command = ['ffmpeg', '-i', filename, f_out]
    subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)


def main():
    # Read metadata file
    for file in os.listdir('coughvid/'):
        f = os.path.join('coughvid/', file)
        f_out = 'coughvid_data/' + file + '.wav'
        if f.endswith('.webm') or f.endswith('.wav') or f.endswith('.ogg'):
            convert(f,f_out)
            






if __name__ == '__main__':
    main()
