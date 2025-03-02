import torch
from flask import Flask, render_template, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
from model.net import word2vec_model
from my_code.function import *
from model import net

# Text-CNN 参数
embedding_size = 50  # 词向量维度
sequence_length = 100  # 句子的最大长度
num_classes = 2  # 正负面评论分类
filter_sizes = [3, 4, 5]
num_filters = 200  # 卷积核数量
num_heads = 10
num_layers = 2
hidden_dim = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model1 = net.TextCNN(embedding_size=embedding_size, num_classes=num_classes,
                     filter_sizes=filter_sizes, num_filters=num_filters)
model1.load_state_dict(torch.load("./model/txtcnn_model.pth",weights_only=True))
model1.eval()
model1.to(device)

model2 = net.TransformerModel(embedding_size=embedding_size, num_classes=num_classes,num_heads=num_heads,
                              num_layers=num_layers,hidden_dim=hidden_dim)
model2.load_state_dict(torch.load("./model/transformer_model.pth",weights_only=True))
model2.eval()
model2.to(device)

model3 = BertForSequenceClassification.from_pretrained("finetuned_bert_model")
tokenizer = BertTokenizer.from_pretrained("finetuned_bert_model")

# 初始化 Flask 应用
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']  # 获取用户输入的文本
    cleaned_input = clean_text(user_input)  # 清理文本
    sentence_vector = get_input([cleaned_input])  # 转换为词向量

    cleaned_texts = [clean_text(text) for text in user_input]
    input2 = np.array([text_to_indices(text, word2vec_model, sequence_length) for text in cleaned_texts])
    input_tensor1 = torch.FloatTensor(np.array(sentence_vector)).to(device)  # 转换为 PyTorch 张量
    input_tensor2 = torch.LongTensor(input2).to(device)
    # 进行三个模型的预测
    with torch.no_grad():
        # Model 1 (Text-CNN)
        output1 = model1(input_tensor1)
        _, predicted1 = torch.max(output1, 1)
        # Model 2 (Transformer)
        output2 = model2(input_tensor2)
        _, predicted2 = torch.max(output2, 1)
        # Model 3 (BERT)
        inputs = tokenizer(cleaned_input, return_tensors='pt', truncation=True, padding=True,
                           max_length=sequence_length)
        with torch.no_grad():
            output3 = model3(**inputs)
        _, predicted3 = torch.max(output3.logits, 1)

    # 投票法，统计三个模型的预测结果
    predictions = [predicted1.item(), predicted2[0].item(), predicted3.item()]
    final_prediction = max(set(predictions), key=predictions.count)  # 多数投票
    # 根据投票结果返回文本分类
    prediction_text = "正面评论" if final_prediction == 1 else "负面评论"
    return jsonify({'prediction': prediction_text})


if __name__ == '__main__':
    app.run(debug=True,port=5001)
