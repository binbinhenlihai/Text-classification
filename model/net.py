import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_size, num_classes, filter_sizes, num_filters):
        super(TextCNN, self).__init__()

        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_size)) for fs in filter_sizes
        ])

        # 全连接层
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x.shape: (batch_size, sequence_length, embedding_size)
        x = x.unsqueeze(1)  # 增加一个维度: (batch_size, 1, sequence_length, embedding_size)

        # 卷积层
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # 池化层
        pool_results = [F.max_pool1d(result, result.size(2)).squeeze(2) for result in conv_results]

        # 拼接池化后的结果
        x = torch.cat(pool_results, 1)

        # 全连接层
        x = self.fc(x)

        return x


class TextCNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_size=50, num_classes=2, filter_sizes=[3, 4, 5], num_filters=100, batch_size=64,
                 epochs=3, learning_rate=0.001):
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # 将数据转换为 PyTorch 张量
        inputs_tensor = torch.FloatTensor(X)
        labels_tensor = torch.LongTensor(y)

        # 创建训练 DataLoader
        train_data = TensorDataset(inputs_tensor, labels_tensor)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # 初始化模型
        self.model = TextCNN(embedding_size=self.embedding_size, num_classes=self.num_classes,
                             filter_sizes=self.filter_sizes, num_filters=self.num_filters)

        # 损失函数和优化器
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 训练过程
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {total_loss / len(train_loader)}")

        return self

    def predict(self, X):
        # 转换测试数据为 PyTorch 张量
        inputs_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word2vec_model = Word2Vec.load("model/word2vec_50dim_large.model")

class TransformerModel(nn.Module):
    def __init__(self, embedding_size, num_classes, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()

        # 将 NumPy 数组转换为 PyTorch tensor
        word2vec_tensor = torch.tensor(word2vec_model.wv.vectors, dtype=torch.float32).to(device)

        # 嵌入层
        self.embedding = nn.Embedding.from_pretrained(word2vec_tensor, freeze=False)  # 使用预训练词向量

        # Transformer模型，设置batch_first=True
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=num_heads,
                                          num_encoder_layers=num_layers, dim_feedforward=hidden_dim,
                                          dropout=dropout, batch_first=True)

        # 分类头
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # x.shape: (batch_size, seq_len)
        x = self.embedding(x)  # 将输入转为嵌入，输出形状为 (batch_size, seq_len, embedding_size)

        # 检查 Transformer 输入的维度是否正确
        assert x.size(-1) == self.transformer.d_model, f"Input dimension {x.size(-1)} does not match d_model {self.transformer.d_model}"

        # Transformer模型
        # 注意：这里的输入 x 和目标 x 是相同的，符合自编码器结构
        transformer_output = self.transformer(x, x)  # 使用自编码器结构，源和目标相同

        # 取Transformer输出的最后一个时间步的输出
        x = transformer_output[:, -1, :]  # 形状变为 (batch_size, embedding_size)

        # 分类层
        x = self.fc(x)  # 直接通过分类层

        return x


