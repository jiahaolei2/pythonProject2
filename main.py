# 这是一个train_loader=Noneort pandas as pd
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from LSTM import LSTM_module
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
# , nrows=100000
def read_Data():
    print('开始读取文件')
    datas = pd.read_csv('NGSIM_Data.csv', usecols=['Vehicle_ID', 'Global_Time', 'Global_X', 'Global_Y', 'v_Class', 'v_Vel','v_Acc' ,  'Location'])  # 读文件
    print('已成功读取文件')
    #  获取i-80数据
    print('正在读取数据')
    datas_i_80 = datas[datas.Location == 'i-80']
    print('正在选择数据')
    datas_i_80_c = datas_i_80[datas_i_80.v_Class == 2]
    datas_i_80_c_2224 = datas_i_80_c[datas_i_80_c.Vehicle_ID == 2224]
    print('正在导出数据')
    datas_i_80_c_2224.sort_values(by='Global_Time', inplace=True)
    datas_i_80_c_2224.to_csv('小型车2224数据_xy.csv')
    print('i-80路段，小型车数据已导出完毕')

def Data_prograss(model, criterion, optimizer):
    # 清理数据
    car1 = pd.read_csv('小型车2224数据.csv')
    # 2.实例化一个转换器
    transfer = StandardScaler()
    # 3.调用transform转换
    car_new = transfer.fit_transform(car1)
    print(car_new)
    # features = car1[['Vehicle_ID', 'Global_Time', 'Global_X', 'Global_Y', 'v_Class', 'v_Vel', 'v_Acc']].values  # 车辆的7个特征
    features = car1[['Global_X', 'Global_Y']].values  # 车辆的7个特征
    label = car1[['Global_X', 'Global_Y']].values
    # 创建序列
    N = 100
    features_seq, label_seq = [], []
    for i in range(N, len(features)):
        features_seq.append(features[i - N: i].astype(float))
        label_seq.append(label[i].astype(float))
    # 将格式转化为tensor
    print(features_seq)
    features_seq, label_seq = torch.FloatTensor(features_seq), torch.FloatTensor(label_seq)
    train_size = int(0.8 * len(features_seq))
    test_size = len(features_seq) - train_size
    train_feature, test_feature = torch.split(features_seq, [train_size, test_size])
    train_label, test_label = torch.split(label_seq, [train_size, test_size])
    train_dataset = TensorDataset(train_feature, train_label)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)  # 批大小 批尺寸 乱序
    print(len(train_loader))
    print(11)
    num_epochs = 600
    for epoch in range(num_epochs):
        for batch_features, batch_labels in train_loader:
            outputs = model(batch_features)
            optimizer.zero_grad()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    # 模型测试
    test_dataset = TensorDataset(test_feature, test_label)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 评估模型
    total_loss = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
    average_test_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {average_test_loss:.4f}')
def Init():
    input_dim = 7  # Vehicle_ID', 'Global_Time', 'Global_X', 'Global_Y', 'v_Class', 'v_Vel', 'v_Acc
    input_dim = 2  # global_x global_y
    hidden_dim = 50  # 隐藏层
    num_layers = 3  # 训练层数
    output_dim = 2  # X, Y
    model = LSTM_module(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    Data_prograss(model, criterion, optimizer)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    #read_Data()
    Init()#初始化模型
    #Data_prograss()




