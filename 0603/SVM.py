from sklearn.svm import SVC  # 导入SVM分类器
from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

# 定义加载和准备数据的函数
def load_and_prepare_data(path, n_samples, label_column):
    df = read_csv(path)
    df = df.head(n_samples)
    X = df.values[:, :-3]
    y = df.values[:, label_column]  # 根据label_column选择标签列
    X = X.astype('float32')
    y = y.astype('float32')
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# 加载训练和测试数据的路径保持不变
path_train = 'E:\\phme2022\\processing_data\\data\\resample\\train1.csv'
path_test = 'E:\\phme2022\\processing_data\\data\\resample\\test1.csv'

# 对每个标签列进行预测
for label_column in [-3, -2, -1]:
    print(f"Predicting for label column: {4+label_column}")
    X_train, y_train = load_and_prepare_data(path_train, 35000, label_column)
    X_test, y_test = load_and_prepare_data(path_test, 15000, label_column)

    # 创建SVM分类器
    model = SVC()  # 使用默认参数

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试数据集的标签
    y_pred = model.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for level {4+label_column}: {accuracy * 100.0:.2f}%")