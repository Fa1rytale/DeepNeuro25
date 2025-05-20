# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:48:26 2025

@author: Denis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Загрузка данных
df = pd.read_csv('dataset_simple.csv')
X = df.iloc[:, :2].values  # Признаки: возраст, доход
y = df.iloc[:, 2].values.reshape(-1, 1)  # Метки: 0 (не купит), 1 (купит)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# 2. Определение нейронной сети
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.layers(X)

# Параметры
inputSize = X_train.shape[1]
hiddenSize = 10  # Уменьшено
outputSize = 1

# Инициализация
net = NNet(inputSize, hiddenSize, outputSize)
lossFn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# 3. Обучение
epochs = 500  # Уменьшено
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = net(X_train)
    loss = lossFn(pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 4. Оценка
with torch.no_grad():
    pred = net(X_test)
    pred_labels = (pred > 0.5).float()
    accuracy = (pred_labels == y_test).float().mean()
    print(f"\nТочность на тестовом наборе: {accuracy.item() * 100:.2f}%")

# 5. Итоговая классификация и ошибки
with torch.no_grad():
    pred = net(X_test)
    rift = (pred.max().item() + pred.min().item()) / 2
    pred_labels = (pred >= rift).float()
    errors = torch.abs(y_test - pred_labels).sum().item() / 2
    print('\nОшибка (количество несовпавших ответов): ')
    print(f'{errors:.0f}')