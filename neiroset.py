import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Инициализация
w1_1, w1_2, w2_1, w2_2, w3_1, w3_2 = [random.uniform(-1, 1) for _ in range(6)]
b1, b2, b3 = [random.uniform(-1, 1) for _ in range(3)]
lr = 0.5

# Данные для XOR
data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]

# Обучение
for epoch in range(10000):
    total_err = 0
    for (x1, x2), target in data:
        # Прямой проход
        h1 = sigmoid(x1*w1_1 + x2*w2_1 + b1)
        h2 = sigmoid(x1*w1_2 + x2*w2_2 + b2)
        out = sigmoid(h1*w3_1 + h2*w3_2 + b3)
        
        # Ошибка
        err = target - out
        total_err += err**2
        
        # Обратное распространение
        d_out = err * sigmoid_derivative(out)
        d_h1 = d_out * w3_1 * sigmoid_derivative(h1)
        d_h2 = d_out * w3_2 * sigmoid_derivative(h2)
        
        # Обновление весов
        w3_1 += lr * d_out * h1
        w3_2 += lr * d_out * h2
        b3 += lr * d_out
        
        w1_1 += lr * d_h1 * x1
        w2_1 += lr * d_h1 * x2
        b1 += lr * d_h1
        
        w1_2 += lr * d_h2 * x1
        w2_2 += lr * d_h2 * x2
        b2 += lr * d_h2
    
    if epoch % 2000 == 0:
        print(f"Эпоха {epoch}, ошибка: {total_err/4:.6f}")

# Проверка
print("\nРезультат:")
for (x1, x2), target in data:
    h1 = sigmoid(x1*w1_1 + x2*w2_1 + b1)
    h2 = sigmoid(x1*w1_2 + x2*w2_2 + b2)
    out = sigmoid(h1*w3_1 + h2*w3_2 + b3)
    print(f"{x1} XOR {x2} = {round(out)} (предсказано: {out:.3f})")
