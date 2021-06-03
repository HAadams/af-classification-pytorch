from model.model import AFModel
from dataset.dataset import AFDataset

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



def test_fn(data_dl, loss_fn):
  model.eval()
  with torch.no_grad():
    for data, labels in data_dl:
      preds = model(data.unsqueeze(1).cuda())
      val_loss = loss_fn(preds.cpu(), labels.squeeze(1).long())

      preds = torch.argmax(preds, dim=-1).cpu()
      f1 = f1_score(preds, labels, average=None)
  model.train()
  return f1, val_loss.detach().item()


  
df = pd.read_json('af_data/af_data_10s.json', orient='records', lines=True)
train, test = train_test_split(df, test_size=0.05, random_state=964607)

model = AFModel(recording_length=len(test.iloc[0].ecg)//300)
optimizer = torch.optim.AdamW(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()


model.train()
model.cuda()
EPOCHS = 5
FOLDS = 5

test_dl = torch.utils.data.DataLoader(
    AFDataset(dataframe=test), batch_size=len(test)
)

fold_count = 1
for train_idx, val_idx in KFold(n_splits=FOLDS).split(train):
  print(f"\nK-Fold: {fold_count}")
  fold_count += 1

  train_df = train.iloc[train_idx]
  val_df = train.iloc[val_idx]

  train_dl = torch.utils.data.DataLoader(
      AFDataset(dataframe=train_df), batch_size=8)
  
  val_dl = torch.utils.data.DataLoader(
      AFDataset(dataframe=val_df), batch_size=len(val_df))
  
  for epoch in range(EPOCHS):
    count = 0
    loss_count = 0
    for vecs, labels in train_dl:
      vecs = vecs.cuda()
      labels = labels.cuda()

      outs = model(vecs.unsqueeze(1))
      loss = loss_fn(outs, labels.squeeze(1).long())

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      count += 1
      loss_count += loss.mean().item()

    f1, loss = test_fn(test_dl, loss_fn)
    print(f'Epoch: {epoch+1}  loss: {loss_count/count} unseen f1: {f1}')
