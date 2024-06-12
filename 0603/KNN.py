from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN分类器

# 定义加载和准备数据的函数
def load_and_prepare_data(path, n_samples):
    df = read_csv(path)
    df = df.head(n_samples)
    X = df.values[:, :-3]
    y = df.values[:, -2]  # 只取label中的第一列
    X = X.astype('float32')
    y = y.astype('float32')
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# 加载训练和测试数据
# path_train = 'E:\phme2022\processing_data\\data\\resample\\datatraining3.csv'
# path_test = 'E:\\phme2022\\processing_data\\data\\resample\\datatesting3.csv'
path_train = 'E:\phme2022\processing_data\\data\\resample\\train1.csv'
path_test = 'E:\\phme2022\\processing_data\\data\\resample\\test1.csv'
X_train, y_train = load_and_prepare_data(path_train, 35000)
X_test, y_test = load_and_prepare_data(path_test, 15000)

# 创建KNN分类器
model = KNeighborsClassifier(n_neighbors=5)  # 使用默认的5个邻居

# 训练模型
model.fit(X_train, y_train)

# 预测测试数据集的标签
y_pred = model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))