import torch.nn as nn
import torch.nn.functional as F

# #79.5ish%
# class ConvolutionalNN(nn.Module):
#     def __init__(self):
#         super(ConvolutionalNN, self).__init__()
#         self.output_size = 1
#
#         self.conv1 = nn.Conv1d(6, 100, 4)
#         self.conv2 = nn.Conv1d(100, 128, 6)
#         self.pool = nn.MaxPool1d(2)
#         self.relu = nn.ReLU()
#         self.avg_pool = nn.AvgPool1d(3)
#         self.conv3 = nn.Conv1d(128, 192, 10, stride=2, padding=1)
#         self.conv4 = nn.Conv1d(192, 256, 5, stride=2, padding=1)
#
#         self.fc1 = nn.Linear(1024, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(64, 26)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))
#         x = self.pool(x)
#         x = x.view(x.shape[0], -1)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         return x

class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.output_size = 1


        self.seq1 = nn.Sequential(
            nn.Conv1d(6, 25, 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(25, 50, 6),
            nn.LeakyReLU(),
            nn.Conv1d(50, 75, 10),
            nn.MaxPool1d(2)
        )
        self.seq2 = nn.Sequential(
            nn.Conv1d(75, 128, 12),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 2),
            nn.LeakyReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 26)
        )


    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x