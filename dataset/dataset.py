
import numpy as np
import torch

class AFDataset(torch.utils.data.Dataset):
  def __init__(self, dataframe):
    self.df = dataframe
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    vals = np.array(self.df.iloc[idx].ecg)
    vals = torch.FloatTensor(vals)
    label = torch.FloatTensor([self.df.iloc[idx].target])
    return vals , label