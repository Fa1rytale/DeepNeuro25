# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 11:46:26 2025

@author: Denis
"""



import torch
from random import choice

# Генерируем тензор с одним случайным числом от 1 до 10
base_tensor = torch.tensor([choice(range(1, 11))], dtype=torch.int64)
print(f"Исходный тензор: {base_tensor.item()}")

# Приводим к float32 и включаем отслеживание градиентов
base_tensor = base_tensor.to(dtype=torch.float32)
base_tensor.requires_grad_(True)

# Возводим в квадрат
exponent = 2
squared_tensor = torch.pow(base_tensor, exponent)
print(f"Тензор в степени {exponent}: {squared_tensor.item()}")

# Умножаем на случайное число
factor = choice(range(1, 11))
multiplied_tensor = squared_tensor * factor
print(f"Тензор, умноженный на {factor}: {multiplied_tensor.item()}")

# Вычисляем экспоненту
exp_tensor = torch.exp(multiplied_tensor)
print(f"Экспонента тензора: {exp_tensor.item()}")

# Выполняем обратное распространение
exp_tensor.backward()
gradient = base_tensor.grad
print(f"Градиент после вычислений: {gradient.item()}")
print(f"Значение градиента: {gradient.item():.6f}")