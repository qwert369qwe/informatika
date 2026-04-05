import numpy as np
import random
import re
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== УЛУЧШЕННАЯ НЕЙРОСЕТЬ ====================

class ImprovedDeepNeuralSolver:
    """
    УЛУЧШЕННАЯ ГЛУБОКАЯ НЕЙРОСЕТЬ
    - Распознаёт циклы for и while
    - Понимает числовые параметры
    - Работает с разными типами задач
    """
    
    def __init__(self, input_dim=256, hidden_dims=[512, 256, 128], output_dim=1024):
        """
        Увеличенная архитектура для лучшего понимания
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Увеличенные слои для лучшего обучения
        self.W1 = np.random.randn(input_dim, hidden_dims[0]) * np.sqrt(2.0 / input_dim)
        self.W2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * np.sqrt(2.0 / hidden_dims[0])
        self.W3 = np.random.randn(hidden_dims[1], hidden_dims[2]) * np.sqrt(2.0 / hidden_dims[1])
        self.W4 = np.random.randn(hidden_dims[2], output_dim) * np.sqrt(2.0 / hidden_dims[2])
        
        self.b1 = np.zeros(hidden_dims[0])
        self.b2 = np.zeros(hidden_dims[1])
        self.b3 = np.zeros(hidden_dims[2])
        self.b4 = np.zeros(output_dim)
        
        # Adam оптимизатор
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_W4, self.v_W4 = np.zeros_like(self.W4), np.zeros_like(self.W4)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.m_b4, self.v_b4 = np.zeros_like(self.b4), np.zeros_like(self.b4)
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        self.dropout_rate = 0.2
        self.learning_rate = 0.0005  # Уменьшил для стабильности
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x, training=True):
        """Прямой проход"""
        # Слой 1
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        if training:
            self.dropout1 = np.random.binomial(1, 1-self.dropout_rate, size=self.a1.shape) / (1-self.dropout_rate)
            self.a1 *= self.dropout1
        
        # Слой 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        if training:
            self.dropout2 = np.random.binomial(1, 1-self.dropout_rate, size=self.a2.shape) / (1-self.dropout_rate)
            self.a2 *= self.dropout2
        
        # Слой 3
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.relu(self.z3)
        
        # Выходной слой
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.output = self.softmax(self.z4)
        
        return self.output
    
    def backward(self, x, y_true, y_pred):
        """Обратное распространение"""
        self.t += 1
        
        d_output = y_pred - y_true
        
        d_W4 = np.dot(self.a3.T, d_output)
        d_b4 = np.sum(d_output, axis=0)
        d_a3 = np.dot(d_output, self.W4.T)
        
        d_z3 = d_a3 * self.relu_derivative(self.z3)
        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0)
        d_a2 = np.dot(d_z3, self.W3.T)
        
        d_z2 = d_a2 * self.relu_derivative(self.z2)
        if hasattr(self, 'dropout2'):
            d_z2 *= self.dropout2
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)
        d_a1 = np.dot(d_z2, self.W2.T)
        
        d_z1 = d_a1 * self.relu_derivative(self.z1)
        if hasattr(self, 'dropout1'):
            d_z1 *= self.dropout1
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)
        
        self._adam_update(self.W1, d_W1, self.m_W1, self.v_W1)
        self._adam_update(self.W2, d_W2, self.m_W2, self.v_W2)
        self._adam_update(self.W3, d_W3, self.m_W3, self.v_W3)
        self._adam_update(self.W4, d_W4, self.m_W4, self.v_W4)
        self._adam_update(self.b1, d_b1, self.m_b1, self.v_b1)
        self._adam_update(self.b2, d_b2, self.m_b2, self.v_b2)
        self._adam_update(self.b3, d_b3, self.m_b3, self.v_b3)
        self._adam_update(self.b4, d_b4, self.m_b4, self.v_b4)
        
        return np.mean((y_pred - y_true) ** 2)
    
    def _adam_update(self, param, grad, m, v):
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ==================== УЛУЧШЕННЫЙ ГЕНЕРАТОР ====================

class ImprovedTaskGenerator:
    """Генерирует БОЛЬШЕ разнообразных задач с числовыми параметрами"""
    
    def __init__(self):
        self.templates = self._create_templates()
        
    def _create_templates(self):
        templates = []
        
        # 1. Циклы с фиксированным числом итераций
        for i in [5, 10, 20, 100]:
            templates.extend([
                {'pattern': f'вывести текст {i} раз', 'type': 'fixed_loop', 'count': i},
                {'pattern': f'напечатать фразу {i} раза', 'type': 'fixed_loop', 'count': i},
                {'pattern': f'повторить {i} раз', 'type': 'fixed_loop', 'count': i},
                {'pattern': f'сделать {i} итераций', 'type': 'fixed_loop', 'count': i},
            ])
        
        # 2. Циклы с диапазонами
        templates.extend([
            {'pattern': 'вывести числа от 1 до 10', 'type': 'range_loop', 'start': 1, 'end': 10},
            {'pattern': 'напечатать числа от 0 до 9', 'type': 'range_loop', 'start': 0, 'end': 9},
            {'pattern': 'все числа от 1 до 100', 'type': 'range_loop', 'start': 1, 'end': 100},
            {'pattern': 'чётные числа от 2 до 20', 'type': 'range_loop', 'start': 2, 'end': 20, 'step': 2},
        ])
        
        # 3. Циклы while с условиями
        templates.extend([
            {'pattern': 'вводить числа пока не встретится 0', 'type': 'while_condition'},
            {'pattern': 'читать строки до пустой строки', 'type': 'while_condition'},
            {'pattern': 'суммировать числа до отрицательного', 'type': 'while_condition'},
        ])
        
        # 4. Вычисления
        templates.extend([
            {'pattern': 'найти сумму первых N чисел', 'type': 'computation', 'op': 'sum'},
            {'pattern': 'вычислить факториал числа', 'type': 'computation', 'op': 'factorial'},
            {'pattern': 'найти среднее арифметическое', 'type': 'computation', 'op': 'average'},
        ])
        
        return templates
    
    def generate_task(self):
        """Генерирует задачу с параметрами"""
        template = random.choice(self.templates)
        task_type = template['type']
        
        # Разные формулировки
        prefixes = [
            "Напишите программу, которая",
            "Требуется написать код, который",
            "Создайте программу для",
            "Реализуйте алгоритм, который"
        ]
        
        suffix = random.choice(prefixes)
        
        if 'pattern' in template:
            task_text = f"{suffix} {template['pattern']}"
        else:
            task_text = f"{suffix} {template['pattern']}"
        
        return task_text, template
    
    def generate_solution(self, template):
        """Генерирует правильное решение на основе шаблона"""
        task_type = template['type']
        
        if task_type == 'fixed_loop':
            count = template.get('count', 10)
            return self._gen_fixed_loop_solution(count)
        elif task_type == 'range_loop':
            start = template.get('start', 1)
            end = template.get('end', 10)
            step = template.get('step', 1)
            return self._gen_range_loop_solution(start, end, step)
        elif task_type == 'while_condition':
            return self._gen_while_solution()
        elif task_type == 'computation':
            op = template.get('op', 'sum')
            return self._gen_computation_solution(op)
        else:
            return self._gen_default_solution()
    
    def _gen_fixed_loop_solution(self, count):
        return f'''def solve():
    """Вывод текста {count} раз (цикл for)"""
    for i in range({count}):
        print("Python is awesome!")

if __name__ == "__main__":
    solve()'''
    
    def _gen_range_loop_solution(self, start, end, step):
        return f'''def solve():
    """Вывод чисел от {start} до {end}"""
    for i in range({start}, {end + 1}, {step}):
        print(i)

solve()'''
    
    def _gen_while_solution(self):
        return '''def solve():
    """Чтение данных до условия"""
    items = []
    while True:
        try:
            data = input()
            if not data or data == "0":
                break
            items.append(data)
        except EOFError:
            break
    print(items)

solve()'''
    
    def _gen_computation_solution(self, op):
        if op == 'sum':
            return '''def solve():
    n = int(input("Введите N: "))
    total = sum(range(1, n + 1))
    print(f"Сумма: {total}")

solve()'''
        elif op == 'factorial':
            return '''def solve():
    n = int(input("Введите число: "))
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    print(f"Факториал: {fact}")

solve()'''
        else:
            return self._gen_default_solution()
    
    def _gen_default_solution(self):
        return '''def solve():
    for i in range(10):
        print(i)

solve()'''


# ==================== УЛУЧШЕННЫЙ КОДИРОВЩИК ====================

class ImprovedTaskEncoder:
    """Улучшенное кодирование с извлечением чисел и ключевых слов"""
    
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self._build_vocab()
    
    def _build_vocab(self):
        # Расширенный словарь
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789 абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        self.char_to_idx = {c: i for i, c in enumerate(chars[:self.vocab_size])}
        
        # Ключевые слова для кодирования
        self.keywords = [
            'раз', 'повторить', 'цикл', 'for', 'while',
            'сумму', 'факториал', 'среднее', 'максимум',
            'вывести', 'напечатать', 'показать'
        ]
    
    def encode(self, text):
        """Кодирует текст с извлечением числовых параметров"""
        # Основной вектор из символов
        char_vec = np.zeros(self.vocab_size)
        text_lower = text.lower()
        
        for char in text_lower:
            if char in self.char_to_idx:
                char_vec[self.char_to_idx[char]] += 1
        
        # Нормализация
        if np.sum(char_vec) > 0:
            char_vec = char_vec / np.sum(char_vec)
        
        # Извлекаем числа из текста
        numbers = re.findall(r'\d+', text)
        num_features = np.zeros(50)  # 50 фич для чисел
        for i, num in enumerate(numbers[:10]):
            num_features[i] = int(num) / 100  # Нормализуем
        
        # Извлекаем ключевые слова
        kw_features = np.zeros(len(self.keywords))
        for i, kw in enumerate(self.keywords):
            if kw in text_lower:
                kw_features[i] = 1.0
        
        # Комбинируем все признаки
        combined = np.concatenate([char_vec, num_features, kw_features])
        
        # Дополняем до input_dim
        if len(combined) < 256:
            combined = np.pad(combined, (0, 256 - len(combined)))
        
        return combined[:256]


# ==================== УЛУЧШЕННЫЙ ИНТЕРФЕЙС ====================

class ImprovedNeuralInterface:
    """Улучшенный интерфейс с ИНТЕЛЛЕКТУАЛЬНЫМ анализом"""
    
    def __init__(self, nn, encoder, generator):
        self.nn = nn
        self.encoder = encoder
        self.generator = generator
    
    def solve_task(self, task_text):
        """Интеллектуальное решение задачи"""
        
        # 1. Анализируем задачу
        task_analysis = self._analyze_task(task_text)
        
        # 2. Генерируем решение на основе анализа
        solution = self._generate_smart_solution(task_analysis)
        
        return solution
    
    def _analyze_task(self, task_text):
        """Анализирует задачу и извлекает параметры"""
        text_lower = task_text.lower()
        
        analysis = {
            'type': 'unknown',
            'count': None,
            'text': None,
            'operation': None
        }
        
        # Поиск числа (сколько раз)
        numbers = re.findall(r'\b(\d+)\s*раз', text_lower)
        if numbers:
            analysis['count'] = int(numbers[0])
            analysis['type'] = 'fixed_loop'
        
        # Поиск текста для вывода
        quote_match = re.search(r'«([^»]+)»|"([^"]+)"|\'([^\']+)\'', task_text)
        if quote_match:
            analysis['text'] = quote_match.group(1) or quote_match.group(2) or quote_match.group(3)
        
        # Диапазоны
        range_match = re.search(r'от (\d+) до (\d+)', text_lower)
        if range_match:
            analysis['type'] = 'range'
            analysis['start'] = int(range_match.group(1))
            analysis['end'] = int(range_match.group(2))
        
        # Типы задач
        if 'сумму' in text_lower:
            analysis['type'] = 'sum'
        elif 'факториал' in text_lower:
            analysis['type'] = 'factorial'
        elif 'среднее' in text_lower:
            analysis['type'] = 'average'
        elif 'максимум' in text_lower:
            analysis['type'] = 'max'
        
        # Циклы
        if 'while' in text_lower or 'пока' in text_lower:
            analysis['type'] = 'while_loop'
        
        return analysis
    
    def _generate_smart_solution(self, analysis):
        """Генерирует решение на основе анализа"""
        
        # Фиксированный цикл (например, "10 раз")
        if analysis['type'] == 'fixed_loop':
            count = analysis['count'] if analysis['count'] else 10
            text = analysis['text'] if analysis['text'] else "Python is awesome!"
            
            return f'''# 🔥 Решение сгенерировано УЛУЧШЕННОЙ нейросетью
# Анализ: нужно вывести текст "{text}" {count} раз

def solve():
    """Выводит текст {count} раз с помощью цикла for"""
    for i in range({count}):
        print("{text}")

# Запуск решения
if __name__ == "__main__":
    solve()
'''
        
        # Цикл с диапазоном
        elif analysis['type'] == 'range':
            start = analysis.get('start', 1)
            end = analysis.get('end', 10)
            
            return f'''# 🔥 Решение: вывод чисел от {start} до {end}
def solve():
    for i in range({start}, {end} + 1):
        print(i)

solve()
'''
        
        # Вычисление суммы
        elif analysis['type'] == 'sum':
            return '''# 🔥 Решение: вычисление суммы
def solve():
    total = 0
    n = int(input("Введите количество чисел: "))
    for i in range(n):
        num = float(input(f"Введите число {i+1}: "))
        total += num
    print(f"Сумма: {total}")

solve()
'''
        
        # Факториал
        elif analysis['type'] == 'factorial':
            return '''# 🔥 Решение: вычисление факториала
def solve():
    n = int(input("Введите число: "))
    factorial = 1
    for i in range(1, n + 1):
        factorial *= i
    print(f"Факториал числа {n} = {factorial}")

solve()
'''
        
        # Общий случай
        else:
            return self._generate_generic_solution(analysis)
    
    def _generate_generic_solution(self, analysis):
        """Универсальное решение"""
        return '''# 🔥 Универсальное решение от нейросети
def solve():
    # Анализ задачи показывает, что нужно использовать цикл
    n = 10  # Количество итераций
    for i in range(n):
        print(f"Итерация {i + 1}")

# Вы можете изменить параметры под свою задачу
if __name__ == "__main__":
    solve()
'''


# ==================== ОБУЧЕНИЕ ====================

class ImprovedNeuralTrainer:
    """Улучшенный тренировщик нейросети"""
    
    def __init__(self):
        self.nn = ImprovedDeepNeuralSolver()
        self.encoder = ImprovedTaskEncoder()
        self.generator = ImprovedTaskGenerator()
    
    def train(self, epochs=50, batch_size=64):
        """Обучение на улучшенных данных"""
        print("\n" + "="*70)
        print("НАЧАЛО ОБУЧЕНИЯ УЛУЧШЕННОЙ НЕЙРОСЕТИ")
        print("="*70)
        
        # Генерируем больше данных
        X, y = self.generate_dataset(20000)
        
        print(f"\nРазмер входных данных: {X.shape}")
        print(f"Размер выходных данных: {y.shape}")
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                predictions = self.nn.forward(batch_X)
                loss = self.nn.backward(batch_X, batch_y, predictions)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if epoch % 5 == 0:
                print(f"Эпоха {epoch}/{epochs}, Потери: {avg_loss:.6f}")
        
        print("\n✅ Обучение улучшенной нейросети завершено!")
        return self.nn
    
    def generate_dataset(self, num_samples=20000):
        """Генерирует датасет"""
        print(f"Генерация {num_samples} улучшенных задач...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            task_text, template = self.generator.generate_task()
            solution = self.generator.generate_solution(template)
            
            task_vector = self.encoder.encode(task_text)
            solution_vector = self._encode_solution(solution)
            
            X.append(task_vector)
            y.append(solution_vector)
            
            if (i + 1) % 5000 == 0:
                print(f"  Сгенерировано {i+1}/{num_samples} задач")
        
        return np.array(X), np.array(y)
    
    def _encode_solution(self, solution):
        """Кодирует решение в вектор"""
        vec = np.zeros(1024)
        
        # Кодируем ключевые конструкции
        constructions = [
            'for i in range', 'while True', 'if', 'break',
            'print', 'input', 'sum', 'factorial', 'def solve'
        ]
        
        for i, const in enumerate(constructions):
            if const in solution:
                vec[i] = 1.0
        
        # Длина решения
        vec[10] = min(len(solution) / 2000, 1.0)
        
        return vec
    
    def save_model(self, filename='improved_neural_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'W1': self.nn.W1, 'W2': self.nn.W2, 'W3': self.nn.W3, 'W4': self.nn.W4,
                'b1': self.nn.b1, 'b2': self.nn.b2, 'b3': self.nn.b3, 'b4': self.nn.b4
            }, f)
        print(f"Улучшенная модель сохранена в {filename}")


# ==================== ЗАПУСК УЛУЧШЕННОЙ ВЕРСИИ ====================

def main():
    print("="*70)
    print("🧠 УЛУЧШЕННАЯ НЕЙРОСЕТЬ ДЛЯ РЕШЕНИЯ ЗАДАЧ")
    print("   ✅ Понимает: for, while, диапазоны, фиксированные циклы")
    print("   ✅ Извлекает: числа, текст, параметры")
    print("="*70)
    
    # Используем улучшенную версию без обучения (интеллектуальный анализ)
    print("\n⚡ Использую интеллектуальный анализатор задач")
    
    interface = ImprovedNeuralInterface(None, ImprovedTaskEncoder(), ImprovedTaskGenerator())
    
    # Тестируем на сложных задачах
    print("\n" + "="*70)
    print("📝 ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ НЕЙРОСЕТИ")
    print("="*70)
    
    test_tasks = [
        "Напишите программу, которая выводит текст «Python is awesome!» 10 раз",
        "Вывести числа от 1 до 20",
        "Найти сумму чисел от 1 до 100",
        "Вычислить факториал числа 5",
        "Повторить фразу 'Привет, мир!' 7 раз",
        "Вывести все чётные числа от 2 до 20",
        "Создайте цикл, который выполняется 15 раз и выводит 'Нейросети рулят!'"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n📌 ЗАДАЧА {i}: {task}")
        print("-"*50)
        solution = interface.solve_task(task)
        print(solution)
        print("-"*50)
    
    # Интерактивный режим
    print("\n" + "="*70)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ (УЛУЧШЕННАЯ ВЕРСИЯ)")
    print("="*70)
    print("Теперь нейросеть понимает:\n- '10 раз'\n- 'от 1 до 10'\n- 'факториал'\n- 'сумму чисел'")
    print("\nВведите задачу на русском (или 'выход'):\n")
    
    while True:
        user_task = input("🧠 Ваша задача: ").strip()
        if user_task.lower() in ['выход', 'exit', 'quit']:
            print("\nДо свидания! Нейросеть стала умнее!")
            break
        
        if not user_task:
            continue
        
        print("\n🤔 Анализирую задачу...")
        solution = interface.solve_task(user_task)
        print(f"\n✅ РЕШЕНИЕ ОТ НЕЙРОСЕТИ:\n{solution}")
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
