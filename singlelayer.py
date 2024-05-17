# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

class CSVDataset(Dataset):
    def __init__(self,path,n_samples):
        df = read_csv(path)
        df = df.head(n_samples)
        self.X = df.values[:,:-3]
        self.y = df.values[:,-3:-2]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx],self.y[idx]]
    

class MLP(Module):
    def __init__(self,n_inputs):
        super(MLP,self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 32)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(32, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    def forward(self,X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        return X        
        
def prepare_data(path,n_samples):
    dataset = CSVDataset(path,n_samples)
    train = dataset
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    return train_dl

# train the model
def train_model(train_dl, model):
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01,momentum=0.7)

    for epoch in range(150):
        losses = []
        loop = tqdm(enumerate(train_dl), total =len(train_dl))
        early_stop_threshold = 0.001
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
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.round()
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    f1 = f1_score(actuals, predictions)
    return f1
 
# make a class prediction for one row of data
def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat
 
# prepare the data
path = 'E:\phme2022\processing_data\data_training.csv'
path_test = 'E:\phme2022\processing_data\data_testing.csv'
train_dl = prepare_data(path,200000)
test_dl = prepare_data(path_test,100000)
model = MLP(53)
# # train the model
train_model(train_dl, model)
# evaluate the model
f1 = evaluate_model(test_dl, model)
print(f1)