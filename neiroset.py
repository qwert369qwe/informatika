import random
import math

# ==================== ФУНКЦИИ АКТИВАЦИИ ====================
def sigmoid(x):
    """Сигмоидная функция активации"""
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    """Производная сигмоиды для обратного распространения"""
    return x * (1 - x)

def relu(x):
    """ReLU функция активации (альтернатива)"""
    return max(0, x)

# ==================== НЕЙРОСЕТЬ ДЛЯ XOR ====================
class NeuralNetworkXOR:
    def __init__(self, hidden_neurons=4, learning_rate=0.8):
        """
        Инициализация нейросети
        hidden_neurons: количество нейронов в скрытом слое
        learning_rate: скорость обучения
        """
        self.lr = learning_rate
        
        # Веса от входного слоя (2 нейрона) к скрытому
        self.w_input_hidden = []
        for i in range(2):  # 2 входа
            neuron_weights = []
            for j in range(hidden_neurons):
                neuron_weights.append(random.uniform(-0.5, 0.5))
            self.w_input_hidden.append(neuron_weights)
        
        # Веса от скрытого слоя к выходному (1 нейрон)
        self.w_hidden_output = []
        for j in range(hidden_neurons):
            self.w_hidden_output.append(random.uniform(-0.5, 0.5))
        
        # Смещения (bias)
        self.bias_hidden = []
        for j in range(hidden_neurons):
            self.bias_hidden.append(random.uniform(-0.5, 0.5))
        
        self.bias_output = random.uniform(-0.5, 0.5)
        
        # Для хранения промежуточных значений
        self.hidden_outputs = []
        self.output = 0
        
    def forward(self, x1, x2):
        """
        Прямой проход: вычисление предсказания
        """
        # Скрытый слой
        self.hidden_outputs = []
        for j in range(len(self.bias_hidden)):
            # Суммируем взвешенные входы + смещение
            total = self.bias_hidden[j]
            total += x1 * self.w_input_hidden[0][j]
            total += x2 * self.w_input_hidden[1][j]
            # Применяем функцию активации
            self.hidden_outputs.append(sigmoid(total))
        
        # Выходной слой
        total_output = self.bias_output
        for j in range(len(self.hidden_outputs)):
            total_output += self.hidden_outputs[j] * self.w_hidden_output[j]
        self.output = sigmoid(total_output)
        
        return self.output
    
    def train(self, training_data, epochs=20000, verbose=True):
        """
        Обучение нейросети
        training_data: список пар ([x1, x2], target)
        epochs: количество эпох обучения
        verbose: выводить ли прогресс
        """
        for epoch in range(epochs):
            total_error = 0
            
            # Перемешиваем данные для лучшего обучения
            shuffled_data = training_data.copy()
            random.shuffle(shuffled_data)
            
            for (x1, x2), target in shuffled_data:
                # Прямой проход
                prediction = self.forward(x1, x2)
                
                # Вычисляем ошибку
                error = target - prediction
                total_error += error ** 2
                
                # === ОБРАТНОЕ РАСПРОСТРАНЕНИЕ ОШИБКИ ===
                
                # Градиент для выходного слоя
                d_output = error * sigmoid_derivative(prediction)
                
                # Градиенты для скрытого слоя
                d_hidden = []
                for j in range(len(self.hidden_outputs)):
                    d = d_output * self.w_hidden_output[j] * sigmoid_derivative(self.hidden_outputs[j])
                    d_hidden.append(d)
                
                # Обновляем веса от скрытого к выходному
                for j in range(len(self.hidden_outputs)):
                    self.w_hidden_output[j] += self.lr * d_output * self.hidden_outputs[j]
                self.bias_output += self.lr * d_output
                
                # Обновляем веса от входного к скрытому
                for i in range(2):  # 2 входа
                    for j in range(len(self.hidden_outputs)):
                        if i == 0:
                            self.w_input_hidden[0][j] += self.lr * d_hidden[j] * x1
                        else:
                            self.w_input_hidden[1][j] += self.lr * d_hidden[j] * x2
                
                # Обновляем смещения скрытого слоя
                for j in range(len(self.hidden_outputs)):
                    self.bias_hidden[j] += self.lr * d_hidden[j]
            
            # Выводим прогресс
            if verbose and (epoch % 2000 == 0 or epoch == epochs - 1):
                avg_error = total_error / len(training_data)
                print(f"Эпоха {epoch:5d}, Ошибка: {avg_error:.6f}")
    
    def predict(self, x1, x2):
        """Предсказание (возвращает 0 или 1)"""
        output = self.forward(x1, x2)
        return 1 if output > 0.5 else 0
    
    def predict_proba(self, x1, x2):
        """Предсказание с вероятностью"""
        return self.forward(x1, x2)

# ==================== УПРОЩЁННАЯ ВЕРСИЯ (для понимания) ====================
def simple_xor_network():
    """
    Минимальная нейросеть для XOR (без классов)
    """
    print("\n" + "="*60)
    print("УПРОЩЁННАЯ ВЕРСИЯ НЕЙРОСЕТИ")
    print("="*60)
    
    # Инициализация весов
    w11, w12 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    w21, w22 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    w31, w32 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    b1, b2, b3 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    lr = 0.8
    
    data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]
    
    print("\nОбучение упрощённой сети...")
    for epoch in range(15000):
        total_err = 0
        for (x1, x2), target in data:
            # Прямой проход
            h1 = sigmoid(x1*w11 + x2*w21 + b1)
            h2 = sigmoid(x1*w12 + x2*w22 + b2)
            out = sigmoid(h1*w31 + h2*w32 + b3)
            
            err = target - out
            total_err += err**2
            
            # Обратное распространение
            d_out = err * sigmoid_derivative(out)
            d_h1 = d_out * w31 * sigmoid_derivative(h1)
            d_h2 = d_out * w32 * sigmoid_derivative(h2)
            
            # Обновление весов
            w31 += lr * d_out * h1
            w32 += lr * d_out * h2
            b3 += lr * d_out
            
            w11 += lr * d_h1 * x1
            w21 += lr * d_h1 * x2
            b1 += lr * d_h1
            
            w12 += lr * d_h2 * x1
            w22 += lr * d_h2 * x2
            b2 += lr * d_h2
        
        if epoch % 3000 == 0:
            print(f"Эпоха {epoch:5d}, Ошибка: {total_err/4:.6f}")
    
    # Результаты
    print("\nРЕЗУЛЬТАТЫ УПРОЩЁННОЙ СЕТИ:")
    correct = 0
    for (x1, x2), target in data:
        h1 = sigmoid(x1*w11 + x2*w21 + b1)
        h2 = sigmoid(x1*w12 + x2*w22 + b2)
        out = sigmoid(h1*w31 + h2*w32 + b3)
        pred = 1 if out > 0.5 else 0
        correct += (pred == target)
        status = "✅" if pred == target else "❌"
        print(f"{status} {x1} XOR {x2} = {pred}  (вероятность: {out:.4f}, должно: {target})")
    
    print(f"\nТочность: {correct}/4 ({correct*25}%)")

# ==================== РАСШИРЕННАЯ ВЕРСИЯ С ТЕСТИРОВАНИЕМ ====================
def extended_test():
    """
    Тестирование нейросети на разных конфигурациях
    """
    print("\n" + "="*60)
    print("РАСШИРЕННОЕ ТЕСТИРОВАНИЕ")
    print("="*60)
    
    # Данные для XOR
    xor_data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]
    
    # Пробуем разное количество нейронов
    configs = [
        {"neurons": 2, "lr": 0.7, "epochs": 15000, "name": "2 нейрона"},
        {"neurons": 4, "lr": 0.8, "epochs": 15000, "name": "4 нейрона"},
        {"neurons": 6, "lr": 0.9, "epochs": 12000, "name": "6 нейронов"}
    ]
    
    best_accuracy = 0
    best_network = None
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        nn = NeuralNetworkXOR(
            hidden_neurons=config["neurons"],
            learning_rate=config["lr"]
        )
        nn.train(xor_data, epochs=config["epochs"], verbose=False)
        
        # Проверяем
        correct = 0
        for (x1, x2), target in xor_data:
            pred = nn.predict(x1, x2)
            if pred == target:
                correct += 1
        
        accuracy = correct / 4 * 100
        print(f"Точность: {correct}/4 ({accuracy}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_network = nn
    
    # Показываем лучшую сеть
    if best_network:
        print(f"\n🏆 ЛУЧШАЯ СЕТЬ (точность {best_accuracy}%):")
        for (x1, x2), target in xor_data:
            proba = best_network.predict_proba(x1, x2)
            pred = best_network.predict(x1, x2)
            print(f"  {x1} XOR {x2} = {pred}  (вероятность: {proba:.4f})")

# ==================== ВИЗУАЛИЗАЦИЯ РЕШЕНИЙ ====================
def visualize_solution(nn):
    """
    Визуализация того, как нейросеть разделяет пространство
    """
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ РАБОТЫ НЕЙРОСЕТИ")
    print("="*60)
    print("\nКарта решений (█ = 1, ░ = 0):")
    print("   x2 →")
    print(" x1↓")
    
    for x1 in [0, 0.25, 0.5, 0.75, 1]:
        row = []
        for x2 in [0, 0.25, 0.5, 0.75, 1]:
            proba = nn.predict_proba(x1, x2)
            row.append("█" if proba > 0.5 else "░")
        print(f" {x1:.2f}  " + " ".join(row))
    
    print("\nЛегенда: █ = предсказано 1, ░ = предсказано 0")

# ==================== ИНТЕРАКТИВНЫЙ РЕЖИМ ====================
def interactive_mode(nn):
    """
    Интерактивный режим: пользователь вводит значения
    """
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*60)
    print("Введите два числа (0 или 1) для XOR")
    print("Или 'выход' для завершения\n")
    
    while True:
        user_input = input("Введите x1 x2: ").strip()
        if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
            break
        
        try:
            parts = user_input.split()
            if len(parts) >= 2:
                x1 = float(parts[0])
                x2 = float(parts[1])
                
                proba = nn.predict_proba(x1, x2)
                pred = nn.predict(x1, x2)
                
                print(f"{x1} XOR {x2} = {pred} (вероятность: {proba:.4f})")
                print(f"{'✅' if pred == (int(x1)^int(x2)) else '⚠️'} {'Правильно' if pred == (int(x1)^int(x2)) else 'Неправильно'}")
            else:
                print("Введите два числа через пробел")
        except ValueError:
            print("Ошибка: введите числа")

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    print("="*60)
    print("    НЕЙРОСЕТЬ ДЛЯ РЕШЕНИЯ ЗАДАЧИ XOR")
    print("    (с использованием циклов for и while)")
    print("="*60)
    
    # Создаём и обучаем основную нейросеть
    print("\n🤖 СОЗДАНИЕ НЕЙРОСЕТИ...")
    nn = NeuralNetworkXOR(hidden_neurons=4, learning_rate=0.8)
    
    print("\n📚 НАЧАЛО ОБУЧЕНИЯ...")
    xor_data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]
    nn.train(xor_data, epochs=20000, verbose=True)
    
    # Проверка на обучающих данных
    print("\n✅ ПРОВЕРКА НА ОБУЧАЮЩИХ ДАННЫХ:")
    correct = 0
    for (x1, x2), target in xor_data:
        proba = nn.predict_proba(x1, x2)
        pred = nn.predict(x1, x2)
        correct += (pred == target)
        status = "✅" if pred == target else "❌"
        print(f"{status} {x1} XOR {x2} = {pred}  (вероятность: {proba:.4f}, должно: {target})")
    
    print(f"\n📊 ТОЧНОСТЬ: {correct}/4 ({correct*25}%)")
    
    # Визуализация
    visualize_solution(nn)
    
    # Запускаем упрощённую версию
    simple_xor_network()
    
    # Расширенное тестирование
    extended_test()
    
    # Интерактивный режим
    interactive_mode(nn)
    
    print("\n" + "="*60)
    print("СПАСИБО ЗА ИСПОЛЬЗОВАНИЕ НЕЙРОСЕТИ!")
    print("="*60)

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    main()
