import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score
from multi_layer_loss import MultiLayerLoss

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)  
        self.data = self.data.head(10000)
        self.scaler = MinMaxScaler()  # 创建 Min-Max 归一化器
        self.features = self.data.iloc[:, :-3]  # 前面的特征
        self.labels = self.data.iloc[:, -3:-2]  # 后面的标签
        self.features = self.scaler.fit_transform(self.features)  # 对特征进行 Min-Max 归一化

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 转换为 PyTorch 张量
        features = torch.Tensor(self.features[idx])
        labels = torch.LongTensor(self.labels.iloc[idx].values)  # 转换为整数形式的类别索引

        return features, labels
    
# 数据集地址
dataset = MyDataset('E:\phme2022\processing_data\data_training.csv') 
# 

# # 创建数据加载器
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ## 打印出来看一下数据
# for i in range(30):
#     features, labels = dataset[i]
#     print("样本", i+1, ":")
#     print("特征:", features)
#     print("标签:", labels)
#     print()
# # 遍历数据加载器
# for inputs, labels in dataloader:
#     # 在这里进行模型训练或其他操作
#     # inputs 是输入特征的张量
#     # labels 是对应的标签的张量
#     pass

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.feature_size = 53
        # self.block0_parameters = [64,32]
        # self.layer1_parameters = [16,2]
        # self.layer2_parameters = [16,3]
        # self.layer3_parameters = [16,4]
        # self.block0 = nn.Sequential(nn.Linear(self.feature_size,self.block0_parameters[0]),
        #                                  nn.ReLU(inplace = True),
        #                                  nn.Linear(self.block0_parameters[0], self.block0_parameters[1]),
        #                                  nn.ReLU(inplace = True))
        # reg模块:从block0中继承信息并进行FC
        # level模块:与之前的level的信息合并后进行FC
        # self.reg1 = nn.Sequential(nn.Linear(self.block0_parameters[-1],self.layer1_parameters[0]),
        #                             nn.ReLU(inplace = True))
        # self.level1 = nn.Sequential(nn.Linear(self.layer1_parameters[0],self.layer1_parameters[1]),
        #                             nn.ReLU(inplace = True))
        
        # self.reg2 = nn.Sequential(nn.Linear(self.block0_parameters[-1],self.layer2_parameters[0]),
        #                             nn.ReLU(inplace = True))
        # self.level2 = nn.Sequential(nn.Linear(self.layer2_parameters[0]+self.layer1_parameters[-1],self.layer2_parameters[1]),
        #                             nn.ReLU(inplace = True))
        
        # self.reg3 = nn.Sequential(nn.Linear(self.block0_parameters[-1],self.layer3_parameters[0]),
        #                             nn.ReLU(inplace = True))
        # self.level3 = nn.Sequential(nn.Linear(self.layer3_parameters[0]+self.layer1_parameters[-1]+self.layer2_parameters[-1],self.layer3_parameters[1]),
        #                             nn.ReLU(inplace = True))
        self.linear1 = nn.Linear(53,32)
        self.linear2 = nn.Linear(32,2)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        # x = self.block0(x)
        # level_1 = self.level1(self.reg1(x))
        # level_2 = self.level2(torch.cat((level_1,self.reg2(x)),dim=1))
        # level_3 = self.level(torch.cat((level_1,level_2,self.reg3(x)),dim=1))
        # output = F.softmax(level_1, dim=1)
        # _, predicted = torch.max(output, 1)
        # return level_1, level_2, level_3
        # return predicted
        y_pred = self.sigmoid(self.linear2(self.relu(self.linear1(x))))
        return y_pred

model =  MyModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)


# 训练循环
epochs = 1000
for epoch in range(epochs):
    running_loss = 0.0
    progress_bar = tqdm(dataloader,desc=f'Epoch {epoch+1}', leave=False)
    for inputs, labels in progress_bar:
        
        outputs = model(inputs)

        # 调整标签形状以匹配输出
        labels = labels.squeeze()

        # 计算损失
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'Loss': running_loss / (len(progress_bar) + 1)})

    # 打印每个 epoch 的损失
    print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader)}")
    # # 在训练集上计算 F1 分数
    # y_true = []
    # y_pred = []
    # model.eval()  # 切换到评估模式
    # with torch.no_grad():
    #     for inputs, labels in dataloader:
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         y_true.extend(labels.tolist())
    #         y_pred.extend(predicted.tolist())

    # f1 = f1_score(y_true, y_pred, average='macro')
    # print(f"Epoch {epoch+1} Training F1 Score: {f1}")

    # model.train()  # 切换回训练模式
print("训练完成")
