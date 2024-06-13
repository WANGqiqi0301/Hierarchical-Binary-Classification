# pytorch mlp for binary classification
import torch
from numpy import vstack
from pandas import read_csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
class CSVDataset(Dataset):
    def __init__(self,path,n_samples):
        df = read_csv(path)
        df = df.head(n_samples)
        self.X = df.values[:,:-3]
        self.y = df.values[:,-2:-1]
        self.X = self.X.astype('float32')
        self.y = torch.from_numpy(self.y).long()
        self.y = self.y.reshape((len(self.y), 1))
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx],self.y[idx][0]]
    

class MLP(Module):
    def __init__(self,n_inputs):
        super(MLP,self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 64)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(64,32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        # second hidden layer
        self.hidden3 = Linear(32, 8)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(8, 3)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = nn.Softmax(dim=1)

    def forward(self,X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        return X        
        
def prepare_data(path,n_samples):
    dataset = CSVDataset(path,n_samples)
    train = dataset
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    return train_dl

class CustomLoss(nn.Module):
    def __init__(self, alpha0, alpha1, alpha2):
        super(CustomLoss, self).__init__()
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, yhat, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(yhat, targets)
        weights = torch.zeros_like(targets).float()
        weights[targets == 0] = self.alpha0
        weights[targets == 1] = self.alpha1
        weights[targets == 2] = self.alpha2
        return (ce_loss * weights).mean()
    
# train the model
def train_model(train_dl, model):
    criterion = nn.CrossEntropyLoss()
    # criterion = CustomLoss(1,1,1)
    optimizer = SGD(model.parameters(), lr=0.007,momentum=0.7)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(120):
        losses = []
        loop = tqdm(enumerate(train_dl), total =len(train_dl))
        early_stop_threshold = 0.01
        for i, (inputs, targets) in loop:
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch+1}/{100}]')
            # loop.set_postfix(loss = loss.item())
        average_loss = sum(losses) / len(losses)
        print(f"Epoch [{epoch+1}/{100}], Average Loss: {average_loss}")
        if average_loss < early_stop_threshold:
            print(f"Early stopping triggered. Loss < {early_stop_threshold}")
            break


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        _, predicted = torch.max(yhat, 1)  # Get class from softmax output
        predictions.extend(predicted.tolist())
        actuals.extend(targets.tolist())
        if i == 0:
            print("First group of predictions:", predicted.tolist())
            print("First group of actual labels:", targets.tolist())
    # calculate confusion matrix
    cm = confusion_matrix(actuals, predictions)
    
    # calculate f1 score for class 1
    tp = cm[1, 1]  # true positive: actual is 1, prediction is 1
    fp = cm[2, 1]  # false positive: actual is 2, prediction is 1
    fn = cm[1, 2]  # false negative: actual is 1, prediction is 2
    # tp = cm[2, 2]  # true positive: actual is 1, prediction is 1
    # fp = cm[0, 2]  # false positive: actual is 2, prediction is 1
    # fn = cm[2, 0]  # false negative: actual is 1, prediction is 2
    f1 = 2*tp / (2*tp + fp + fn)
    return f1
 
# make a class prediction for one row of data
def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat
 
# prepare the data
# path = 'E:\phme2022\processing_data\data_training.csv'
path = 'E:\phme2022\processing_data\\resampledatatraining.csv'
# path = 'E:\phme2022\processing_data\shortdatatraining.csv'
# path_test = 'E:\phme2022\processing_data\data_testing1.csv'
path_test = 'E:\phme2022\processing_data\\resampledatatesting.csv'
train_dl = prepare_data(path,77158)
test_dl = prepare_data(path_test,31815)
model = MLP(53)
# # train the model
train_model(train_dl, model)
# evaluate the model
f1_train = evaluate_model(train_dl, model)
f1_test = evaluate_model(test_dl, model)

print(f"Training score: {f1_train}")
print(f"Testing score: {f1_test}")