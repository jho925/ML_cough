import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

def open_audio(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
    
    return (sig, sr)


def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)
