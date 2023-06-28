import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextCNN(nn.Module):
    # embed_num: 词表词数
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout, attention):
        super(TextCNN, self).__init__()

        Ci = 1
        Co = kernel_num
        self.attention = attention

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        # 自注意力层
        if self.attention:
            self.self_attention = nn.ModuleList([nn.Linear(Co * len(kernel_sizes), embed_dim)]*3)
            # self.q = nn.Linear(Co * len(kernel_sizes), embed_dim)
            # self.k = nn.Linear(Co * len(kernel_sizes), embed_dim)
            # self.v = nn.Linear(Co * len(kernel_sizes), embed_dim)
            self.fc = nn.Linear(embed_dim, class_num)
        else:
            self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.to(device)  # (N, Ci, max_length, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1)  # (N, Co * len(kernel_sizes))
        x = self.dropout(x) # (N, Co * len(kernel_sizes))

        if self.attention:
            Q, K, V = [fc(x) for fc in self.self_attention]
            # Q = self.q(x)   # (N, embed_dim)
            # K = self.k(x)   # (N, embed_dim)
            # V = self.v(x)   # (N, embed_dim)
            S = torch.matmul(Q, K.transpose(-2, -1)) # (N, N) 注意力分数
            A = F.softmax(S, dim=-1)# 注意力权重
            x = torch.matmul(A, V)# (N, embed_dim)

        x = self.fc(x)
        prob = F.softmax(x, dim=1)  # (N, class_num)
        return prob


class LSTM(nn.Module):
    def __init__(self, embed_dim, class_num, bidirectional, dropout, attention, hidden_dim=128, num_layers=1):
        super().__init__()
        self.attention = attention
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional,
                            dropout=dropout)
        if self.attention:
            self.w1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
            self.w2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, class_num)

    def forward(self, x):
        # x的形状为(N, max_length, embedding_dim)
        x = x.permute(1, 0, 2)
        output, hidden_cell = self.lstm(x)
        # output的形状为(sequence_length, N, hidden_dim)
        # hidden_cell:((1,N,hidden_dim),(1,N,hidden_dim))

        if self.attention:
            # 注意力分数
            attention_scores = self.w2(torch.tanh(self.w1(output))).squeeze(2) # max_length, N
            # 注意力权重
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2) # max_length, N, 1

            output = torch.sum(output * attention_weights, dim=1) # max_length, hidden_dim
        hidden = hidden_cell[0]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) if self.lstm.bidirectional else hidden[-1, :, :])
        # N, hidden_dim
        output = self.fc(hidden)
        return output  # 最后一层全连接层，输出形状为(N, class_num)


class MLP(nn.Module):
    def __init__(self, input_dim, class_num, attention, hidden_dim=128):
        super().__init__()
        self.attention = attention
        # input_dim:max_document_length*embedding_dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        if self.attention:
            self.attention_fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        # x形状为(N, max_length, dimension)
        x = x.view(x.size(0), -1) # (N, max_length*dimension)
        output = self.fc1(x) # N, hidden_dim
        output = self.relu(output) # N, hidden_dim
        if self.attention:
            output = F.softmax(self.attention_fc(output), dim=0) # N, hidden_dim softmax函数将隐藏层输出的每个元素变成了一个[0,1]之间的值，这些值的总和为1。这些值可以看作是输入的不同部分在注意力中的重要程度
        output = self.fc2(output) # N, class_num
        return output
