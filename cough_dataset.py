from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import pad_trunc, spectro_gram, open_audio, resample, rechannel


class cough_dataloader(Dataset):
  def __init__(self, df):
    self.df = df
    self.duration = 4000
    self.channel = 2
    self.sr = 44100
  
  def __len__(self):
    return len(self.df)    
    
  def __getitem__(self, idx):
    audio_file =  self.df.loc[idx, 'path']
    is_cough = self.df.loc[idx, 'is_cough']

    aud = open_audio(audio_file)

  
    reaud = resample(aud, self.sr)
    rechan = rechannel(reaud, self.channel)
    aud_dur = pad_trunc(rechan, self.duration)
    aud_sgram = spectro_gram(aud_dur, n_mels=64, n_fft=1024, hop_len=None)

    return aud_sgram, is_cough