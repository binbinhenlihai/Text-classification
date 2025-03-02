import re

import nltk
import numpy as np
from nltk.corpus import stopwords

# 下载 NLTK 停用词库
nltk.download('stopwords')  # 确保停用词库已下载
stop_words = set(stopwords.words('english'))

def load_word_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(i) for i in parts[1:]], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors

# 假设训练好的词向量存储在 word2vec.txt 文件中
word_vectors = load_word_vectors('txt/word2vec_vectors_50d.txt')

# 定义文本清理函数
def clean_text(text):
    # 转小写
    text = text.lower()

    # 去除标点符号和非字母字符（可选保留数字）
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 按空格分词
    words = text.split()

    # 去除停用词
    cleaned_words = [word for word in words if word not in stop_words]

    # 重新组合为字符串
    cleaned_text = ' '.join(cleaned_words)

    # 去除多余的空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text
# Text-CNN 参数
embedding_size = 50  # 词向量维度
sequence_length = 100  # 句子的最大长度
num_classes = 2  # 正负面评论分类
filter_sizes = [3, 4, 5]
num_filters = 200  # 卷积核数量
def get_input(texts):
    # 创建输入词向量
    inputs = []
    for sen in texts:
        sentence_vector = []
        for word in sen.split():
            # 使用读取的词向量字典来获取每个单词的词向量
            if word in word_vectors:
                sentence_vector.append(word_vectors[word])  # 获取词向量
            else:
                sentence_vector.append(np.zeros(embedding_size))  # 如果词不在词汇表中，使用零向量
        if len(sentence_vector) < sequence_length:
            sentence_vector.extend([np.zeros(embedding_size)] * (sequence_length - len(sentence_vector)))
        else:
            sentence_vector = sentence_vector[:sequence_length]
        inputs.append(np.array(sentence_vector))
    return inputs


def text_to_indices(text, model, sequence_length=100):
    sentence_indices = []
    for word in text.split():
        if word in model.wv:
            # 获取词的索引
            sentence_indices.append(model.wv.key_to_index[word])
        else:
            sentence_indices.append(0)  # 如果词不在词汇表中，使用0（通常代表未知词）

    # 填充或截断到固定长度
    if len(sentence_indices) < sequence_length:
        sentence_indices.extend([0] * (sequence_length - len(sentence_indices)))
    else:
        sentence_indices = sentence_indices[:sequence_length]

    return np.array(sentence_indices)
