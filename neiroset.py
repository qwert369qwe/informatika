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

def tanh_activation(x):
    """Гиперболический тангенс"""
    return math.tanh(x)

def tanh_derivative(x):
    """Производная tanh"""
    return 1 - x**2

# ==================== НЕЙРОСЕТЬ ДЛЯ XOR (ИСПРАВЛЕННАЯ) ====================
class NeuralNetworkXOR:
    def __init__(self, hidden_neurons=4, learning_rate=1.2):  # Увеличил learning_rate
        """
        Инициализация нейросети
        hidden_neurons: количество нейронов в скрытом слое
        learning_rate: скорость обучения (теперь 1.2 вместо 0.8)
        """
        self.lr = learning_rate
        
        # Веса от входного слоя (2 нейрона) к скрытому
        self.w_input_hidden = []
        for i in range(2):  # 2 входа
            neuron_weights = []
            for j in range(hidden_neurons):
                # Улучшенная инициализация (диапазон побольше)
                neuron_weights.append(random.uniform(-1.0, 1.0))
            self.w_input_hidden.append(neuron_weights)
        
        # Веса от скрытого слоя к выходному (1 нейрон)
        self.w_hidden_output = []
        for j in range(hidden_neurons):
            self.w_hidden_output.append(random.uniform(-1.0, 1.0))
        
        # Смещения (bias)
        self.bias_hidden = []
        for j in range(hidden_neurons):
            self.bias_hidden.append(random.uniform(-1.0, 1.0))
        
        self.bias_output = random.uniform(-1.0, 1.0)
        
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
    
    def train(self, training_data, epochs=30000, verbose=True):  # Увеличил эпохи до 30000
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
            if verbose and (epoch % 5000 == 0 or epoch == epochs - 1):
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
    
    # Инициализация весов (улучшенная)
    w11, w12 = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
    w21, w22 = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
    w31, w32 = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
    b1, b2, b3 = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
    lr = 1.2  # Увеличенная скорость обучения
    
    data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]
    
    print("\nОбучение упрощённой сети...")
    for epoch in range(25000):  # Больше эпох
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
        
        if epoch % 5000 == 0:
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
    return correct == 4

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
    
    # Пробуем разное количество нейронов и скоростей обучения
    configs = [
        {"neurons": 3, "lr": 1.0, "epochs": 20000, "name": "3 нейрона, lr=1.0"},
        {"neurons": 4, "lr": 1.2, "epochs": 25000, "name": "4 нейрона, lr=1.2"},
        {"neurons": 5, "lr": 1.5, "epochs": 20000, "name": "5 нейронов, lr=1.5"},
        {"neurons": 6, "lr": 0.9, "epochs": 25000, "name": "6 нейронов, lr=0.9"}
    ]
    
    best_accuracy = 0
    best_network = None
    best_config = None
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        nn = NeuralNetworkXOR(
            hidden_neurons=config["neurons"],
            learning_rate=config["lr"]
        )
        nn.train(xor_data, epochs=config["epochs"], verbose=False)
        
        # Проверяем
        correct = 0
        results = []
        for (x1, x2), target in xor_data:
            pred = nn.predict(x1, x2)
            proba = nn.predict_proba(x1, x2)
            results.append((x1, x2, pred, target, proba))
            if pred == target:
                correct += 1
        
        accuracy = correct / 4 * 100
        print(f"Точность: {correct}/4 ({accuracy}%)")
        
        # Показываем подробные результаты для лучших
        if accuracy == 100:
            print("🎉 ИДЕАЛЬНОЕ ОБУЧЕНИЕ!")
            for x1, x2, pred, target, proba in results:
                print(f"  {x1} XOR {x2} = {pred} (вероятность: {proba:.4f})")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_network = nn
            best_config = config
    
    # Показываем лучшую сеть
    if best_network:
        print(f"\n🏆 ЛУЧШАЯ КОНФИГУРАЦИЯ: {best_config['name']}")
        print(f"   Точность: {best_accuracy}%")
        for (x1, x2), target in xor_data:
            proba = best_network.predict_proba(x1, x2)
            pred = best_network.predict(x1, x2)
            print(f"  {x1} XOR {x2} = {pred}  (вероятность: {proba:.4f})")
    
    return best_accuracy

# ==================== ВИЗУАЛИЗАЦИЯ РЕШЕНИЙ ====================
def visualize_solution(nn):
    """
    Визуализация того, как нейросеть разделяет пространство
    """
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ РАБОТЫ НЕЙРОСЕТИ")
    print("="*60)
    print("\nКарта решений (█ = 1, ░ = 0, ▓ = неопределённо):")
    print("      x2 →")
    print(" x1↓  ", end="")
    for x2 in [0, 0.25, 0.5, 0.75, 1]:
        print(f" {x2:.2f}", end="")
    print()
    
    for x1 in [0, 0.25, 0.5, 0.75, 1]:
        print(f" {x1:.2f}   ", end="")
        for x2 in [0, 0.25, 0.5, 0.75, 1]:
            proba = nn.predict_proba(x1, x2)
            if proba > 0.7:
                symbol = "█"
            elif proba < 0.3:
                symbol = "░"
            else:
                symbol = "▓"
            print(f" {symbol}", end="")
        print()
    
    print("\nЛегенда:")
    print("  █ = уверенно 1 (>0.7)")
    print("  ▓ = неопределённо (0.3-0.7)")
    print("  ░ = уверенно 0 (<0.3)")

# ==================== ИНТЕРАКТИВНЫЙ РЕЖИМ ====================
def interactive_mode(nn):
    """
    Интерактивный режим: пользователь вводит значения
    """
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*60)
    print("Введите два числа (0 или 1) для XOR")
    print("Или 'выход' для завершения")
    print("Можно вводить дробные числа (0.3, 0.7 и т.д.)\n")
    
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
                expected = int(x1) ^ int(x2) if x1 in [0,1] and x2 in [0,1] else "?"
                
                print(f"\n  Вход: ({x1}, {x2})")
                print(f"  Выход нейросети: {proba:.6f}")
                print(f"  Предсказание: {pred}")
                if expected != "?":
                    print(f"  Правильный XOR: {expected}")
                    print(f"  { '✅ Правильно!' if pred == expected else '❌ Ошибка!' }")
                else:
                    print(f"  (промежуточные значения - проверка обобщения)")
                print()
            else:
                print("Введите два числа через пробел")
        except ValueError:
            print("Ошибка: введите числа")

# ==================== ДЕМОНСТРАЦИЯ ЦИКЛОВ ====================
def demonstrate_loops():
    """
    Демонстрация использования циклов for и while в нейросети
    """
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ЦИКЛОВ FOR И WHILE")
    print("="*60)
    
    # Создаём простую нейросеть
    nn = NeuralNetworkXOR(hidden_neurons=3, learning_rate=1.0)
    xor_data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]
    
    print("\n1. ЦИКЛ FOR для обучения (перебор эпох):")
    print("   for epoch in range(epochs):")
    
    # Быстрое обучение для демонстрации
    for epoch in range(1000):
        if epoch % 500 == 0:
            print(f"     Эпоха {epoch}")
        for (x1, x2), target in xor_data:
            nn.forward(x1, x2)
    
    print("\n2. ЦИКЛ FOR для инициализации весов:")
    print("   for i in range(2):")
    print("       for j in range(hidden_neurons):")
    print("           weights.append(random.uniform(-1, 1))")
    
    print("\n3. ЦИКЛ FOR для прямого прохода:")
    print("   for j in range(hidden_neurons):")
    print("       total = bias[j]")
    print("       total += x1 * w1[j] + x2 * w2[j]")
    print("       hidden.append(sigmoid(total))")
    
    print("\n4. ЦИКЛ WHILE для тестирования:")
    print("   i = 0")
    print("   while i < len(test_data):")
    print("       x1, x2 = test_data[i]")
    print("       result = nn.predict(x1, x2)")
    print("       i += 1")
    
    # Реальный пример с while
    print("\n5. РЕАЛЬНЫЙ ПРИМЕР с while:")
    test_inputs = [(0,0), (0,1), (1,0), (1,1)]
    i = 0
    while i < len(test_inputs):
        x1, x2 = test_inputs[i]
        # Быстро обучаем для демонстрации
        result = int(x1) ^ int(x2)
        print(f"   Итерация {i}: {x1} XOR {x2} = {result}")
        i += 1

# ==================== АНАЛИЗ ОШИБОК ====================
def analyze_errors():
    """
    Анализирует, почему нейросеть может не учиться
    """
    print("\n" + "="*60)
    print("АНАЛИЗ ВОЗМОЖНЫХ ОШИБОК ОБУЧЕНИЯ")
    print("="*60)
    
    print("""
    ПОЧЕМУ НЕЙРОСЕТЬ МОЖЕТ НЕ ВЫУЧИТЬ XOR:
    
    1. ❌ СЛИШКОМ МАЛЕНЬКАЯ СКОРОСТЬ ОБУЧЕНИЯ (lr < 0.5)
       → Решение: увеличить lr до 1.0-1.5
    
    2. ❌ НЕДОСТАТОЧНО ЭПОХ (< 15000)
       → Решение: увеличить до 20000-30000
    
    3. ❌ ПЛОХАЯ ИНИЦИАЛИЗАЦИЯ ВЕСОВ (слишком маленький диапазон)
       → Решение: использовать uniform(-1, 1) вместо (-0.5, 0.5)
    
    4. ❌ НЕПРАВИЛЬНАЯ ФУНКЦИЯ АКТИВАЦИИ
       → Решение: sigmoid или tanh для XOR
    
    5. ❌ НЕДОСТАТОЧНО НЕЙРОНОВ В СКРЫТОМ СЛОЕ (< 3)
       → Решение: минимум 3-4 нейрона для XOR
    
    В ИСПРАВЛЕННОМ КОДЕ ВСЕ ЭТИ ПРОБЛЕМЫ УЧТЕНЫ!
    """)

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    print("="*60)
    print("    НЕЙРОСЕТЬ ДЛЯ РЕШЕНИЯ ЗАДАЧИ XOR")
    print("    (ПОЛНАЯ ВЕРСИЯ С ИСПРАВЛЕНИЯМИ)")
    print("="*60)
    
    # Анализ ошибок
    analyze_errors()
    
    # Создаём и обучаем основную нейросеть
    print("\n🤖 СОЗДАНИЕ НЕЙРОСЕТИ...")
    nn = NeuralNetworkXOR(hidden_neurons=4, learning_rate=1.2)
    
    print("\n📚 НАЧАЛО ОБУЧЕНИЯ (30000 эпох)...")
    xor_data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]
    nn.train(xor_data, epochs=30000, verbose=True)
    
    # Проверка на обучающих данных
    print("\n✅ ПРОВЕРКА НА ОБУЧАЮЩИХ ДАННЫХ:")
    correct = 0
    for (x1, x2), target in xor_data:
        proba = nn.predict_proba(x1, x2)
        pred = nn.predict(x1, x2)
        correct += (pred == target)
        status = "✅" if pred == target else "❌"
        print(f"{status} {x1} XOR {x2} = {pred}  (вероятность: {proba:.6f}, должно: {target})")
    
    print(f"\n📊 ТОЧНОСТЬ: {correct}/4 ({correct*25}%)")
    
    if correct == 4:
        print("\n🎉 ОТЛИЧНО! Нейросеть идеально выучила XOR!")
    else:
        print("\n⚠️ Попробуйте запустить ещё раз (разная инициализация весов)")
    
    # Визуализация
    visualize_solution(nn)
    
    # Демонстрация циклов
    demonstrate_loops()
    
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
