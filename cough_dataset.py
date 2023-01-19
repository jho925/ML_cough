from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import pad_trunc, spectro_gram, open_audio


class cough_dataloader(Dataset):
  def __init__(self, df):
    self.df = df
    self.duration = 5387
  
  def __len__(self):
    return len(self.df)    
    
  def __getitem__(self, idx):
    audio_file =  self.df.loc[idx, 'path']
    is_cough = self.df.loc[idx, 'is_cough']

    aud = open_audio(audio_file)

    aud_dur = pad_trunc(aud, self.duration)
    aud_sgram = spectro_gram(aud_dur, n_mels=64, n_fft=1024, hop_len=None)

    if is_cough == True:
      return aud_sgram, 1
    else:
      return aud_sgram, 0