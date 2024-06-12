# pytorch mlp for binary classification
import random
import numpy as np
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from multi_layer_loss import MultiLayerLoss
from sklearn.metrics import confusion_matrix
class CSVDataset(Dataset):
    def __init__(self,path,n_samples):
        df = read_csv(path)
        df = df.head(n_samples)
        self.X = df.values[:,:-3]
        self.y = df.values[:,-3:-1]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 2))
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx],self.y[idx]]
    

class Multi_LP(nn.Module):
    def __init__(self, input_dim):
        super(Multi_LP, self).__init__()
        block0_para = [64,32]
        level1_para = [16,2]
        level2_para = [16,3]
        level3_para = [16,4]
        self.block0_fc1 = nn.Linear(input_dim, block0_para[0])
        self.block0_fc2 = nn.Linear(block0_para[0], block0_para[1])
        
        self.level1_fc1 = nn.Linear(block0_para[-1], level1_para[0])
        self.level1_fc2 = nn.Linear(level1_para[0], level1_para[1])
        
        self.level2_fc1 = nn.Linear(block0_para[-1]+level1_para[-1], level2_para[0])
        self.level2_fc2 = nn.Linear(level2_para[0], level2_para[1])
        
        self.level3_fc1 = nn.Linear(block0_para[-1]+level1_para[-1]+level2_para[-1], level3_para[0])
        self.level3_fc2 = nn.Linear(level3_para[0], level3_para[1])

    def forward(self, x):
        block0 = F.relu(self.block0_fc2(F.relu(self.block0_fc1(x))))
        
        level1 = F.relu(self.level1_fc2(F.relu(self.level1_fc1(block0))))
        
        level2_input = torch.cat([block0, level1], dim=1)
        level2 = F.relu(self.level2_fc2(F.relu(self.level2_fc1(level2_input))))
        
        # level3_input = torch.cat([block0, level1, level2], dim=1)
        # level3 = F.relu(self.level3_fc2(F.relu(self.level3_fc1(level3_input))))
        return level1, level2        
        # return level1, level2, level3
        
        
        
def prepare_data(path,n_samples):
    dataset = CSVDataset(path,n_samples)
    train = dataset
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    return train_dl

# train the model
def train_model(train_dl, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.7)
    MLPloss = MultiLayerLoss(total_level=2, alpha=1, beta=0.8, p_loss=3,device='cpu')
    train_epoch_loss = []
    train_level1_acc = []
    train_level2_acc = []
    train_level3_acc = []
    for epoch in range(130):
        epoch_loss = []
        epoch_level1_acc = []
        epoch_level2_acc = []
        # epoch_level3_acc = []
        loop = tqdm(enumerate(train_dl), total =len(train_dl))
        early_stop_threshold =-0.1
        for i, (inputs, targets) in loop:
            optimizer.zero_grad()
            level1_pred, level2_pred = model(inputs)
            # level1_pred, level2_pred, level3_pred = model(inputs)
            dloss = MLPloss.calculate_dloss([level1_pred, level2_pred], [targets[:,0], targets[:,1]])
            lloss = MLPloss.calculate_lloss([level1_pred, level2_pred], [targets[:,0], targets[:,1]])
            # dloss = MLPloss.calculate_dloss([level1_pred, level2_pred, level3_pred], [targets[:,0], targets[:,1], targets[:,2]])
            # lloss = MLPloss.calculate_lloss([level1_pred, level2_pred, level3_pred], [targets[:,0], targets[:,1], targets[:,2]])
            total_loss = lloss + dloss
            total_loss.backward()
            optimizer.step()
            epoch_loss.append(total_loss.item())
            epoch_level1_acc.append(accuracy_score(targets[:,0].numpy(), torch.argmax(level1_pred, dim=1).detach().numpy()))
            epoch_level2_acc.append(accuracy_score(targets[:,1].numpy(), torch.argmax(level2_pred, dim=1).detach().numpy()))
            # epoch_level3_acc.append(accuracy_score(targets[:,2], level3_pred))
            loop.set_description(f'Epoch [{epoch+1}/{100}]')
            # loop.set_postfix(loss = loss.item())
        train_epoch_loss.append(sum(epoch_loss)/(i+1))
        train_level1_acc.append(sum(epoch_level1_acc)/(i+1))
        train_level2_acc.append(sum(epoch_level2_acc)/(i+1))
        # train_level3_acc.append(sum(epoch_level3_acc)/(i+1))

        print(f" Average Loss: {sum(epoch_loss)/(i+1)}")
        print(f'Epoch [{epoch+1}/{100}],Level 1 Accuracy: {sum(epoch_level1_acc)/(i+1)}')
        print(f'Epoch [{epoch+1}/{100}],Level 2 Accuracy: {sum(epoch_level2_acc)/(i+1)}')
        # print(f'Epoch [{epoch+1}/{100}],Level 3 Accuracy: {sum(epoch_level3_acc)/(i+1)}')
        if sum(epoch_loss)/(i+1) < early_stop_threshold:
            print(f"Early stopping triggered. Loss < {early_stop_threshold}")
            break

def evaluate_model(test_dl, model):
    model.eval()  # 设置模型为评估模式

    # level1_preds, level2_preds, level3_preds = [], [], []
    # level1_targets, level2_targets, level3_targets = [], [], []
    level1_preds, level2_preds = [], []
    level1_targets, level2_targets = [], []
    with torch.no_grad():  # 禁用梯度计算
        for data in test_dl:
            inputs, targets = data
            outputs = model(inputs)

            # 将预测结果和目标添加到列表中
            level1_preds.append(outputs[0].argmax(dim=1).numpy())
            level2_preds.append(outputs[1].argmax(dim=1).numpy())
            # level3_preds.append(outputs[2].argmax(dim=1))

            level1_targets.append(targets[:,0])
            level2_targets.append(targets[:,1])
            # level3_targets.append(targets[2])

    # 将列表转换为numpy数组
    level1_preds = np.concatenate(level1_preds)
    level2_preds = np.concatenate(level2_preds)
    # level3_preds = np.concatenate(level3_preds)

    level1_targets = np.concatenate(level1_targets)
    level2_targets = np.concatenate(level2_targets)
    # level3_targets = np.concatenate(level3_targets)

    # 计算F1分数
    epsilon = 1e-7
    f1 = 2*np.sum((level1_targets == 0) & (level1_preds == 0)) / (2*np.sum((level1_targets == 0) & (level1_preds == 0)) + np.sum((level1_targets == 1) & (level1_preds == 0)) + np.sum((level1_targets == 1) & (level1_preds == 1))+ epsilon)
    f2 = 2*np.sum((level2_targets == 1) & (level2_preds == 1)) / (2*np.sum((level2_targets == 1) & (level2_preds == 1)) + np.sum((level2_targets == 2) & (level2_preds == 1)) + np.sum((level2_targets == 1) & (level2_preds == 2))+ epsilon)
    # f3 = 2*np.sum((level3_targets == 3) & (level3_preds == 3)) / (2*np.sum((level3_targets == 3) & (level3_preds == 3)) + np.sum((level3_targets == 3) & (level3_preds == 2)) + np.sum((level3_targets == 2) & (level3_preds == 3)))

    # return f1, f2, f3
    return f1, f2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(50)  # 使用你选择的任何数字作为种子
 
# prepare the data
path = 'E:\phme2022\processing_data\\resampledatatraining.csv'
path_test = 'E:\phme2022\processing_data\\resampledatatesting.csv'
# path_test = 'E:\phme2022\processing_data\\data_testing1.csv'
train_dl = prepare_data(path,70000)
test_dl = prepare_data(path_test,30000)
model = Multi_LP(53)
# # train the model
train_model(train_dl, model)
# evaluate the model
f1_train,f2_train = evaluate_model(train_dl, model)
f1_test, f2_test = evaluate_model(test_dl, model)
print(f"Training score (level 1): {f1_train}")
print(f"Training score (level 2): {f2_train}")
print(f"Testing score (level 1): {f1_test}")
print(f"Tetsing score (level 2): {f2_test}")
# print(f"Testing score: {f1_test}")
