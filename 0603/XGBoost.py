import xgboost as xgb
from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

# 修改后的加载和准备数据的函数，增加了target_column参数
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

# 加载训练和测试数据的路径保持不变
path_train = 'E:\\phme2022\\processing_data\\data\\resample\\train1.csv'
path_test = 'E:\\phme2022\\processing_data\\data\\resample\\test1.csv'

# 对每个目标列进行预测和评估
for target_column, num_class in zip([-3, -2, -1], [2, 3, 4]):
    # 加载并准备数据
    X_train, y_train = load_and_prepare_data(path_train, 35000, target_column)
    X_test, y_test = load_and_prepare_data(path_test, 15000, target_column)

    # 创建XGBoost分类器，根据目标列的类别数量设置objective
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_class, use_label_encoder=False)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试数据集的标签
    y_pred = model.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for level {4+target_column}: {accuracy * 100.0:.2f}%")