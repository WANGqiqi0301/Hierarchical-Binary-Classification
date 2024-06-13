from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN分类器

# 定义加载和准备数据的函数
def load_and_prepare_data(path, n_samples, target_column):
    df = read_csv(path)
    df = df.head(n_samples)
    X = df.values[:, :-3]  # 保持特征选择不变
    y = df.values[:, target_column]  # 根据传入的列索引选择目标列
    X = X.astype('float32')
    y = y.astype('float32')
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# 加载训练和测试数据的路径
path_train = 'E:\\phme2022\\processing_data\\data\\resample\\train1.csv'
path_test = 'E:\\phme2022\\processing_data\\data\\resample\\test1.csv'

# 创建KNN分类器
model = KNeighborsClassifier(n_neighbors=5)  # 使用默认的5个邻居

# 对每个目标列进行预测和评估
# for target_column in [-3,-2,-1]:
#     # 加载并准备数据
#     X_train, y_train = load_and_prepare_data(path_train, 35000, target_column)
#     X_test, y_test = load_and_prepare_data(path_test, 15000, target_column)

#     # 训练模型
#     model.fit(X_train, y_train)

#     # 预测测试数据集的标签
#     y_pred = model.predict(X_test)

#     # 评估模型性能
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy for level {target_column+4}: {accuracy * 100.0:.2f}%")

for target_column, n_neighbors in zip([-3, -2, -1], [6,5,4]):
    # 加载并准备数据
    X_train, y_train = load_and_prepare_data(path_train, 35000, target_column)
    X_test, y_test = load_and_prepare_data(path_test, 15000, target_column)

    # 创建XGBoost分类器，根据目标列的类别数量设置objective
    model = KNeighborsClassifier(n_neighbors=n_neighbors)  # 使用默认的5个邻居


    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试数据集的标签
    y_pred = model.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for level {4+target_column}: {accuracy * 100.0:.2f}%")