# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:24:28 2025

@author: Denis
"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Загружаем данные
df = pd.read_csv('data.csv')
print("Первые строки данных:\n", df.head())

# Извлекаем признаки и целевую переменную
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)  # Метки: 1 для Iris-setosa, -1 для других
X = df.iloc[:, [0, 1, 2, 3]].values  # Берем четыре признака

# Преобразуем данные в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Создаем линейный слой: 4 входных признака, 1 выход
linear = nn.Linear(4, 1)
# Добавляем активацию tanh для масштабирования выходов в диапазон [-1, 1]
model = nn.Sequential(linear, nn.Tanh())

# Выводим начальные веса и смещение
print("Начальные веса (w): ", linear.weight)
print("Начальное смещение (b): ", linear.bias)

# Определяем функцию потерь и оптимизатор
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Увеличиваем lr

# Начальный прямой проход и вычисление ошибки
y_pred = model(X_tensor)
loss = loss_fn(y_pred, y_tensor)
print("Начальная ошибка: ", loss.item())

# Обучение модели
max_iterations = 100
i = 1
while i <= max_iterations:
    y_pred = model(X_tensor)
    loss = loss_fn(y_pred, y_tensor)
    print(f"Ошибка на {i}-й итерации: {loss.item()}")

    # Обратное распространение и обновление весов
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if loss.item() < 0.01:  # Уменьшаем порог остановки
        break
    i += 1

# Порог классификации теперь фиксированный (так как выходы в [-1, 1])
threshold = 0.0  # Порог 0, так как tanh масштабирует в [-1, 1]
print(f"Порог для классификации: {threshold}")

# Интерактивный ввод для предсказания
print("\nКлассификация нового цветка:")
while True:
    print("Введите четыре параметра цветка:")
    try:
        p1 = float(input("Признак 1: "))
        p2 = float(input("Признак 2: "))
        p3 = float(input("Признак 3: "))
        p4 = float(input("Признак 4: "))
        new_flower = torch.tensor([p1, p2, p3, p4], dtype=torch.float32)
        
        # Прогноз
        prediction = model(new_flower)
        print("\nВид цветка: ", end="")
        if prediction.item() >= threshold:
            print("Iris-setosa")
        else:
            print("Iris-versicolor")
        
        # Запрос на завершение
        choice = input("\nЗавершить работу? (введите y или Yes): ")
        if choice.lower() in ["y", "yes"]:
            break
    except ValueError:
        print("Ошибка: введите числовые значения для признаков!")