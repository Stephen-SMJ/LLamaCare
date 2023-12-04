import torch
from torch import nn
from transformers import LlamaForCausalLM
from param_dict import max_seq_length

class ClassificationHead1(nn.Module):  #先用pooling降维

    def __init__(self, input_dim=5120, num_classes=3): #[batch_size, 9,728,000]
        super(ClassificationHead1, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=8)
        self.avg_pool = nn.AvgPool1d(kernel_size=8)
        # print("input_dim:",input_dim)
        self.fc1 = nn.Linear(input_dim//64, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64,16)
        self.fc5 = nn.Linear(16,4)
        self.fc6 = nn.Linear(4, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        return x

class ClassificationHead2(nn.Module): #不用pooling降维，直接做映射

    def __init__(self, input_dim=5120, num_classes=3): #[batch_size, 9,728,000]
        super(ClassificationHead2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8,4)
        self.fc6 = nn.Linear(4,num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        return x

class ClassificationHead3(nn.Module): #只取最后的一部分neural

    def __init__(self, input_dim=5120, num_classes=3): #[batch_size, 9,728,000]
        super(ClassificationHead3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8,4)
        self.fc6 = nn.Linear(4,num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        return x

class ClassificationHead4(nn.Module):  #先用pooling降维

    def __init__(self, input_dim=5120, num_classes=3): #[batch_size, 9,728,000]
        super(ClassificationHead4, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=8)
        self.avg_pool = nn.AvgPool1d(kernel_size=8)
        # print("input_dim:",input_dim)
        self.fc1 = nn.Linear(input_dim, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x

class LlamaWithClassifier(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaWithClassifier, self).__init__(config)
        self.classification_head = ClassificationHead(input_dim=max_seq_length*4096, num_classes=3)

    # def forward(self, input_ids, labels=None, **kwargs):
    #     outputs = super(LlamaWithClassifier, self).forward(input_ids, **kwargs)
    #
    #     return outputs