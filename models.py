import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLOBFeatureExtractor(nn.Module):
    def __init__(self, num_levels=10, num_features=4):
        super(DeepLOBFeatureExtractor, self).__init__()
        # Input shape: (Batch, 1, Seq, num_levels * num_features)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        # x shape: (Batch, 1, Seq, 40)
        x = F.relu(self.conv1(x)) # (Batch, 16, Seq, 20)
        x = F.relu(self.conv2(x)) # (Batch, 16, Seq, 10)
        x = F.relu(self.conv3(x)) # (Batch, 32, Seq, 10)
        x = F.relu(self.conv4(x)) # (Batch, 32, Seq, 10)
        return x

class DeepLOBModel(nn.Module):
    def __init__(self, num_levels=10, num_features=4, seq_length=50, num_classes=3):
        super(DeepLOBModel, self).__init__()
        self.extractor = DeepLOBFeatureExtractor(num_levels, num_features)
        self.lstm = nn.LSTM(input_size=32 * 10, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (Batch, Seq, Features)
        batch_size, seq_len, _ = x.shape
        x = x.unsqueeze(1) # Add channel dim
        x = self.extractor(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, -1)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])
