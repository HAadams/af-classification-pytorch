from torch import nn

class AFModel(nn.Module):
  def __init__(self, recording_length:int):
      super().__init__()
      assert recording_length in [6, 10, 30]
      if recording_length in [6, 10]:
        filer = (6)
      else:
        filer = (32)

      self.c1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=filer)
      self.bn1 = nn.BatchNorm1d(64)

      self.c2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=filer)
      self.bn2 = nn.BatchNorm1d(64)

      self.c3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=filer)
      self.bn3 = nn.BatchNorm1d(64)

      self.c4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=filer)
      self.bn4 = nn.BatchNorm1d(64)

      self.c5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=filer)
      self.bn5 = nn.BatchNorm1d(64)

      self.c6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=filer)
      self.bn6 = nn.BatchNorm1d(64)

      self.c7 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=filer)
      self.bn7 = nn.BatchNorm1d(128)

      self.c8 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=filer)
      self.bn8 = nn.BatchNorm1d(128)

      self.c9 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=filer)
      self.bn9 = nn.BatchNorm1d(128)

      self.c10 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=filer)
      self.bn10 = nn.BatchNorm1d(128)

      self.c11 = nn.Conv1d(in_channels=128, out_channels=196, kernel_size=filer)
      self.bn11 = nn.BatchNorm1d(196)

      self.c12 = nn.Conv1d(in_channels=196, out_channels=196, kernel_size=filer)
      self.bn12 = nn.BatchNorm1d(196)

      self.c13 = nn.Conv1d(in_channels=196, out_channels=196, kernel_size=filer)
      self.bn13 = nn.BatchNorm1d(196)

      if recording_length == 6:
        self.output = nn.Linear(in_features=784, out_features=4) # 6s ds
      elif recording_length == 10:
        self.output = nn.Linear(in_features=2548, out_features=4) # 10s ds
      elif recording_length == 30:
        self.output = nn.Linear(in_features=1764, out_features=4) # 30s ds
    
  def forward(self, data):
    relu = nn.ReLU()
    pool = nn.MaxPool1d(kernel_size=2, stride=2)
    dropout = nn.Dropout(0.1)

    data = self.c1(data)
    data = relu(self.bn1(data))
    data = dropout(data)
    data = pool(data)

    data = self.c2(data)
    data = relu(self.bn2(data))

    data = self.c3(data)
    data = relu(self.bn3(data))
    data = dropout(data)
    data = pool(data)

    data = self.c4(data)
    data = relu(self.bn4(data))

    data = self.c5(data)
    data = relu(self.bn5(data))
    data = dropout(data)
    data = pool(data)

    data = self.c6(data)
    data = relu(self.bn6(data))

    data = self.c7(data)
    data = relu(self.bn7(data))
    data = dropout(data)
    data = pool(data)

    data = self.c8(data)
    data = relu(self.bn8(data))

    data = self.c9(data)
    data = relu(self.bn9(data))
    data = dropout(data)
    data = pool(data)

    data = self.c10(data)
    data = relu(self.bn10(data))

    data = self.c11(data)
    data = relu(self.bn11(data))
    data = dropout(data)
    data = pool(data)

    data = self.c12(data)
    data = relu(self.bn12(data))

    data = self.c13(data)
    data = relu(self.bn13(data))
    data = dropout(data)
    data = pool(data)

    data = nn.Flatten()(data)
    # print(data.shape)
    return nn.Softmax(dim=-1)(self.output(data))
