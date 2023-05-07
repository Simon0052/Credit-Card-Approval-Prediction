import pandas as pd
import torch
from torch import nn, optim
import torchvision
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from  sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('creditcard/application_record.csv', header = 0, sep=",") #txt has 3 space so sep="   "
record = pd.read_csv('creditcard/credit_record.csv', header = 0, sep=",") #txt has 3 space so sep="   "
# df_sorted = df.sort_values('ID')
# print(data)
data.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Car','FLAG_OWN_REALTY':'Reality',
                        'CNT_CHILDREN':'ChldNo','AMT_INCOME_TOTAL':'inc',
                        'NAME_EDUCATION_TYPE':'edutp','NAME_FAMILY_STATUS':'famtp',
                        'NAME_HOUSING_TYPE':'houtp','FLAG_EMAIL':'email',
                        'NAME_INCOME_TYPE':'inctp','FLAG_WORK_PHONE':'wkphone',
                        'FLAG_PHONE':'phone','CNT_FAM_MEMBERS':'famsize',
                        'OCCUPATION_TYPE':'occyp'
                        },inplace=True)

def convert_dummy(df, feature,rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest],axis=1,inplace=True)
    df.drop([feature],axis=1,inplace=True)
    df=df.join(pos)
    return df

def get_category(df, col, binsnum, labels, qcut = False):
    if qcut:
        localdf = pd.qcut(df[col], q = binsnum, labels = labels) # quantile cut
    else:
        localdf = pd.cut(df[col], bins = binsnum, labels = labels) # equal-length cut
        
    localdf = pd.DataFrame(localdf)
    name = 'gp' + '_' + col
    localdf[name] = localdf[col]
    df = df.join(localdf[name])
    df[name] = df[name].astype(object)
    return df

# find all users' account open month.
begin_month=pd.DataFrame(record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
begin_month=begin_month.rename(columns={'MONTHS_BALANCE':'begin_month'}) 
new_data=pd.merge(data,begin_month,how="left",on="ID") #merge to record data

record['dep_value'] = None
record['dep_value'][record['STATUS'] == '2' ]='Yes' 
record['dep_value'][record['STATUS'] == '3' ]='Yes' 
record['dep_value'][record['STATUS'] == '4' ]='Yes' 
record['dep_value'][record['STATUS'] == '5' ]='Yes' 
# print(record)
# print(record['dep_value'].value_counts())

cpunt=record.groupby('ID').count()
cpunt['dep_value'][cpunt['dep_value'] > 0]='Yes' 
cpunt['dep_value'][cpunt['dep_value'] == 0]='No' 
cpunt = cpunt[['dep_value']]

new_data=pd.merge(new_data,cpunt,how='inner',on='ID')
new_data['target']=new_data['dep_value']
new_data.loc[new_data['target']=='Yes','target']=1
new_data.loc[new_data['target']=='No','target']=0


new_data['Gender'] = new_data['Gender'].replace(['F','M'],[0,1])
new_data['Car'] = new_data['Car'].replace(['N','Y'],[0,1])
new_data['Reality'] = new_data['Reality'].replace(['N','Y'],[0,1])
# labels, levels = pd.factorize(new_data['inc'])
# new_data['inc'] = labels
new_data['Age']=-(new_data['DAYS_BIRTH'])//365
new_data['worktm']=-(new_data['DAYS_EMPLOYED'])//365	
new_data['famsize']=new_data['famsize'].astype(int)

labels, levels = pd.factorize(new_data['inctp'])
new_data['inctp'] = labels
# new_data = convert_dummy(new_data,'inctp')
# print(new_data['inctp'])
# new_data.loc[(new_data['occyp']=='Cleaning staff') | (new_data['occyp']=='Cooking staff') | (new_data['occyp']=='Drivers') | (new_data['occyp']=='Laborers') | (new_data['occyp']=='Low-skill Laborers') | (new_data['occyp']=='Security staff') | (new_data['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'
# new_data.loc[(new_data['occyp']=='Accountants') | (new_data['occyp']=='Core staff') | (new_data['occyp']=='HR staff') | (new_data['occyp']=='Medicine staff') | (new_data['occyp']=='Private service staff') | (new_data['occyp']=='Realty agents') | (new_data['occyp']=='Sales staff') | (new_data['occyp']=='Secretaries'),'occyp']='officewk'
# new_data.loc[(new_data['occyp']=='Managers') | (new_data['occyp']=='High skill tech staff') | (new_data['occyp']=='IT staff'),'occyp']='hightecwk'
labels, levels = pd.factorize(new_data['occyp'])
new_data['occyp'] = labels
# new_data = convert_dummy(new_data,'occyp')

labels, levels = pd.factorize(new_data['houtp'])
new_data['houtp'] = labels
# new_data = convert_dummy(new_data,'houtp')

# new_data.loc[new_data['edutp']=='Academic degree','edutp']='Higher education'
labels, levels = pd.factorize(new_data['edutp'])
new_data['edutp'] = labels
# new_data = convert_dummy(new_data,'edutp')

labels, levels = pd.factorize(new_data['famtp'])
new_data['famtp'] = labels
# new_data = convert_dummy(new_data,'famtp')

# print(new_data.columns)
# print(new_data['inctp'].value_counts())
# n_splits=1 表示分成1份， 測試集大小为20%， random_state=42 保證每次訓練模型產生的測試集和訓練集都不變
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(new_data, new_data['inctp']):  # 根据收入類别進行分類
    train_data = new_data.loc[train_index]
    test_data = new_data.loc[test_index]

# 查看测试集中收入类别比例分布
# print('分層抽樣的收入類別比例分布：')
# print(test_data['inctp'].value_counts() / len(test_data))

# print('原數據中收入類別的比例分布：')
# print(new_data['inctp'].value_counts() / len(new_data))



x_train = train_data[['Gender', 'Car', 'Reality', 'ChldNo', 'inc', 'inctp', 'edutp','famtp', 'houtp', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'occyp', 'famsize']].to_numpy(dtype='float32')
x_test =  test_data[['Gender', 'Car', 'Reality', 'ChldNo', 'inc', 'inctp', 'edutp','famtp', 'houtp', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'occyp', 'famsize']].to_numpy(dtype='float32')   
# print(x)
# print(type(x))
# print(x.shape)
# print(x.dtype)

# x = x.T
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
# print(x)
label = train_data['target'].to_numpy()
# label = torch.from_numpy (label)
y_train = pd.get_dummies(train_data['target']).to_numpy()
y_test = pd.get_dummies(test_data['target']).to_numpy()
# y = new_data['target'].to_numpy()
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
# print(y)
# print(y.shape)
# print(y.dtype)


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear (input_dim, 600) 
        self.linear2 = nn.Linear (600, 400) 
        self.linear3 = nn.Linear (400, 200) 
        self.linear4 = nn.Linear (200, out_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
    
        return x

#device = torch.device("mps")
model = MLP(13,2)#.to(device)
# print(model)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = 1e-4)
# print(y_train.shape)

#pred = model(x.float())



def train():
    train_loss = 0
    train_acc = 0
    correct = 0 
    count = 0
    model.train()
    #x = x.to(device)
    #y = y.to(device)
    pred = model(x_train)
    loss = loss_f(pred, torch.max(y_train,1)[1])

    opt.zero_grad()
    loss.backward()
    opt.step()

    train_loss = loss.item()
    
    _, y = pred.max(1)
    #print(y)
    correct = (y == torch.max(y_train,1)[1]).sum().item() #max抓出最大值
    #correct = int(correct)
    #train_acc += correct
    train_acc = correct / train_data.shape[0]
    # print(correct)

    return train_loss, train_acc

def test():
    test_loss = 0
    test_acc = 0
    correct = 0 
    count = 0
    model.eval()
    
    pred = model(x_test)
    loss = loss_f(pred, torch.max(y_test,1)[1])
    
    test_loss = loss.item()

    _, y = pred.max(1)
    correct = (y == torch.max(y_test,1)[1]).sum().item() #max抓出最大值
    correct = int(correct)
    test_acc += correct
    test_acc = test_acc / test_data.shape[0]
    
    return test_loss, test_acc 


train_losses = []
train_acces = []
test_losses = []
test_acces = []

epoch = 20

for i in range(epoch):
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    
    train_losses.append(train_loss)
    train_acces.append(train_acc)
    test_losses.append(test_loss)
    test_acces.append(test_acc)

    print("epoch:{}, train_loss:{:.6f}, train_acc:{:.2f}, test_loss:{:.6f}, test_acc:{:.2f}".format(i+1, train_loss, train_acc, test_loss, test_acc))
    
torch.save(model.state_dict(), 'creditcard_approval_model.pt') 

plt.figure()
plt.plot(train_losses, 'r', label = 'train_loss')
plt.plot(test_losses, 'b', label = 'test_loss')
plt.plot(train_acces, 'g', label = 'train_acc')
plt.plot(test_acces, 'y', label = 'test_acc')
plt.legend(loc = 'upper right')
plt.savefig('loss.png')
plt.show()

# y = pd.get_dummies(df[2]).to_numpy()
# y = torch.from_numpy(y)

# print(df)"""