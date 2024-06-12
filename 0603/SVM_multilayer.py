import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from torch.utils.data import Dataset
import numpy as np

# 定义CSVDataset类
class CSVDataset(Dataset):
    def __init__(self, path, n_samples):
        df = read_csv(path)
        df = df.head(n_samples)
        self.X = df.values[:, :-3].astype('float32')
        self.y = df.values[:, -3:].astype('float32')
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 准备数据的函数
def prepare_data(path, n_samples):
    dataset = CSVDataset(path, n_samples)
    return dataset.X, dataset.y

# 加载数据
X_train, y_train = prepare_data('E:\\phme2022\\processing_data\\data\\resample\\datatraining3.csv', 100000)
X_test, y_test = prepare_data('E:\\phme2022\\processing_data\\data\\resample\\datatesting3.csv', 300000)

# 分离标签
y_train_1, y_train_2, y_train_3 = y_train[:, 0], y_train[:, 1], y_train[:, 2]
y_test_1, y_test_2, y_test_3 = y_test[:, 0], y_test[:, 1], y_test[:, 2]

# 使用SVC训练模型
models = [SVC() for _ in range(3)]
models[0].fit(X_train, y_train_1)
models[1].fit(X_train, y_train_2)
models[2].fit(X_train, y_train_3)

# 预测
predictions = [model.predict(X_test) for model in models]

# 计算准确率
accuracy1 = accuracy_score(y_test_1, predictions[0])

correct_indices_layer1 = y_test_1 == predictions[0]
correct_indices_layer2 = y_test_2 == predictions[1]
correct_indices_for_layer2 = correct_indices_layer1 & correct_indices_layer2
accuracy2 = np.sum(correct_indices_for_layer2) / len(y_test_1)

correct_indices_layer3 = y_test_3 == predictions[2]
correct_indices_for_layer3 = correct_indices_for_layer2 & correct_indices_layer3
accuracy3 = np.sum(correct_indices_for_layer3) / len(y_test_1)

print(f'Layer 1 Accuracy: {accuracy1}')
print(f'Layer 2 Accuracy: {accuracy2}')
print(f'Layer 3 Accuracy: {accuracy3}')