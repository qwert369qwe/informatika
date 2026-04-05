
import numpy as np
import random
import re
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

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

        # НОВЫЕ ЗАДАЧИ
        # 5. Печать символьных паттернов
        templates.append({'pattern': 'печати следующей последовательности символов', 'type': 'char_pattern'})
        
        # 6. Повторение предложения N раз
        templates.append({'pattern': 'повторяет данное предложение нужное количество раз', 'type': 'repeat_sentence'})

        # 7. Звёздный прямоугольник
        templates.append({'pattern': 'печатает звёздный прямоугольник', 'type': 'star_rectangle'})

        # 8. Печать строки 10 раз с нумерацией
        templates.append({'pattern': 'выводит 10 строк, пронумерованных от 0 до 9 каждая, с указанной строкой текста', 'type': 'numbered_string_print'})

        # 9. Квадрат числа от 0 до n
        templates.append({'pattern': 'квадрат числа <текущее число> равен <квадрат текущего числа>', 'type': 'squares_up_to_n'})

        # 10. Звёздный треугольник
        templates.append({'pattern': 'выводит звёздный треугольник', 'type': 'star_triangle'})

        # 11. Прогноз популяции организмов
        templates.append({'pattern': 'предсказывает размер популяции организмов', 'type': 'population_growth'})

        # 12. Последовательность слов до «КОНЕЦ»
        templates.append({'pattern': 'выводит члены данной последовательности', 'type': 'sequence_until_keyword', 'keyword': 'КОНЕЦ'})
        templates.append({'pattern': 'выводит члены данной последовательности', 'type': 'sequence_until_keyword', 'keyword': 'КОНЕЦ или конец'})
        templates.append({'pattern': 'выводит общее количество членов данной последовательности', 'type': 'sequence_until_multiple_keywords', 'keywords': ['стоп', 'хватит', 'достаточно']})

        # 13. Последовательность чисел, делящихся на 7
        templates.append({'pattern': 'выводит члены данной последовательности', 'type': 'sequence_divisible_by_7'})

        # 14. Сумма последовательности чисел до отрицательного
        templates.append({'pattern': 'выводит сумму всех членов данной последовательности', 'type': 'sequence_sum_until_negative'})

        # 15. Количество пятерок в последовательности оценок
        templates.append({'pattern': 'выводит количество пятерок', 'type': 'sequence_count_fives'})

        return templates
    
    def generate_task(self):
        """Генерирует задачу с параметрами"""
        template = random.choice(self.templates)
        
        # Разные формулировки
        prefixes = [
            "Напишите программу, которая",
            "Требуется написать код, который",
            "Создайте программу для",
            "Реализуйте алгоритм, который"
        ]
        
        suffix = random.choice(prefixes)
        
        if "pattern" in template:
            task_text = f'{suffix} {template["pattern"]}'
        else:
            task_text = f'{suffix} {template["pattern"]}'
        
        return task_text, template
    
    def generate_solution(self, template):
        """Генерирует правильное решение на основе шаблона"""
        task_type = template["type"]
        
        if task_type == "fixed_loop":
            count = template.get("count", 10)
            return self._gen_fixed_loop_solution(count)
        elif task_type == "range_loop":
            start = template.get("start", 1)
            end = template.get("end", 10)
            step = template.get("step", 1)
            return self._gen_range_loop_solution(start, end, step)
        elif task_type == "while_condition":
            return self._gen_while_solution()
        elif task_type == "computation":
            op = template.get("op", "sum")
            return self._gen_computation_solution(op)
        elif task_type == "char_pattern":
            return self._gen_char_pattern_solution()
        elif task_type == "repeat_sentence":
            return self._gen_repeat_sentence_solution()
        elif task_type == "star_rectangle":
            return self._gen_star_rectangle_solution()
        elif task_type == "numbered_string_print":
            return self._gen_numbered_string_print_solution()
        elif task_type == "squares_up_to_n":
            return self._gen_squares_up_to_n_solution()
        elif task_type == "star_triangle":
            return self._gen_star_triangle_solution()
        elif task_type == "population_growth":
            return self._gen_population_growth_solution()
        elif task_type == "sequence_until_keyword":
            keyword = template.get("keyword", "КОНЕЦ")
            return self._gen_sequence_until_keyword_solution(keyword)
        elif task_type == "sequence_until_multiple_keywords":
            keywords = template.get("keywords", ["стоп", "хватит", "достаточно"])
            return self._gen_sequence_until_multiple_keywords_solution(keywords)
        elif task_type == "sequence_divisible_by_7":
            return self._gen_sequence_divisible_by_7_solution()
        elif task_type == "sequence_sum_until_negative":
            return self._gen_sequence_sum_until_negative_solution()
        elif task_type == "sequence_count_fives":
            return self._gen_sequence_count_fives_solution()
        else:
            return self._gen_default_solution()

    def _gen_fixed_loop_solution(self, count):
        return f"""def solve():
    for i in range({count}):
        print("Python is awesome!")

solve()"""
    
    def _gen_range_loop_solution(self, start, end, step):
        return f"""def solve():
    for i in range({start}, {end + 1}, {step}):
        print(i)

solve()"""
    
    def _gen_while_solution(self):
        return """def solve():
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

solve()"""
    
    def _gen_computation_solution(self, op):
        if op == "sum":
            return """def solve():
    n = int(input())
    total = sum(range(1, n + 1))
    print(f"Сумма: {total}")

solve()"""
        elif op == "factorial":
            return """def solve():
    n = int(input())
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    print(f"Факториал: {fact}")

solve()"""
        else:
            return self._gen_default_solution()

    def _gen_char_pattern_solution(self):
        return """def solve():
    for _ in range(6):
        print("AAA")
    for _ in range(5):
        print("BBBB")
    print("E")
    for _ in range(9):
        print("TTTTT")
    print("G")

solve()"""

    def _gen_repeat_sentence_solution(self):
        return """def solve():
    sentence = input()
    count = int(input())
    for _ in range(count):
        print(sentence)

solve()"""

    def _gen_star_rectangle_solution(self):
        return """def solve():
    n = int(input())
    for _ in range(n):
        print("*" * 19)

solve()"""

    def _gen_numbered_string_print_solution(self):
        return """def solve():
    text = input()
    for i in range(10):
        print(f"{i} {text}")

solve()"""

    def _gen_squares_up_to_n_solution(self):
        return """def solve():
    n = int(input())
    for i in range(n + 1):
        print(f"Квадрат числа {i} равен {i*i}")

solve()"""

    def _gen_star_triangle_solution(self):
        return """def solve():
    n = int(input())
    for i in range(n + 1):
        print("*" * i)

solve()"""

    def _gen_population_growth_solution(self):
        return """def solve():
    start_pop = int(input())
    growth_rate = float(input())
    days = int(input())

    current_pop = start_pop
    for day in range(days):
        current_pop *= (1 + growth_rate)
        print(f"День {day+1}: {int(current_pop)}")

solve()"""

    def _gen_sequence_until_keyword_solution(self, keyword):
        return f"""def solve():
    items = []
    while True:
        item = input()
        if item == \"{keyword}\":
            break
        items.append(item)
    for item in items:
        print(item)

solve()"""

    def _gen_sequence_until_multiple_keywords_solution(self, keywords):
        keywords_str = ", ".join([f'\"{k}\"' for k in keywords])
        return f"""def solve():
    count = 0
    while True:
        item = input()
        if item in [{keywords_str}]:
            break
        count += 1
    print(f"Количество членов: {{count}}")

solve()"""

    def _gen_sequence_divisible_by_7_solution(self):
        return """def solve():
    numbers = []
    while True:
        num_str = input()
        if not num_str:
            break
        num = int(num_str)
        if num % 7 == 0:
            numbers.append(num)
    for num in numbers:
        print(num)

solve()"""

    def _gen_sequence_sum_until_negative_solution(self):
        return """def solve():
    total_sum = 0
    while True:
        num = int(input())
        if num < 0:
            break
        total_sum += num
    print(f"Сумма: {{total_sum}}")

solve()"""

    def _gen_sequence_count_fives_solution(self):
        return """def solve():
    count_fives = 0
    while True:
        grade_str = input()
        if not grade_str:
            break
        grade = int(grade_str)
        if grade == 5:
            count_fives += 1
    print(f"Количество пятерок: {{count_fives}}")

solve()"""

    def _gen_default_solution(self):
        return """def solve():
    print("Hello, World!")

solve()"""




# ==================== УЛУЧШЕННЫЙ КОДИРОВЩИК ====================

class ImprovedTaskEncoder:
    """Улучшенный кодировщик задач, учитывающий больше параметров"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.next_idx = 0
        self._build_vocab_from_keywords()
        
    def _build_vocab_from_keywords(self):
        keywords = [
            # Общие
            "программа", "написать", "код", "реализовать", "алгоритм",
            "вывести", "напечатать", "повторить", "сделать", "итераций",
            "числа", "текст", "строки", "символы", "последовательность",
            "ввод", "вывод", "пока", "до", "если", "равен", "больше", "меньше",
            "сумма", "факториал", "среднее", "количество", "размер",
            
            # Циклы
            "for", "while", "цикл", "диапазон", "от", "до", "шаг",
            
            # Условия
            "условие", "встретится", "пустой", "отрицательный", "конец",
            "стоп", "хватит", "достаточно", "делится", "неположительный",
            
            # Параметры
            "раз", "число", "N", "M", "P", "дней", "высота", "ширина",
            
            # Специфичные для задач
            "звездный", "прямоугольник", "треугольник", "квадрат", "оценка",
            "пятерок", "организм", "популяция", "процент", "среднесуточное",
            "AAA", "BBBB", "E", "TTTTT", "G", # Для символьных паттернов
            "строка", "предложение", "повторений", # Для повторения строк
            "высота", "ширина", # Для прямоугольника
            "пронумерованных", # Для нумерованных строк
            "квадрат", # Для квадратов чисел
            "катет", # Для треугольника
            "стартовое", "увеличение", # Для популяции
            "целых", "делящихся", "любое", "отрицательное", "неположительное", "оценка", "ученика", "пятерок"
        ]
        for word in keywords:
            self._add_word(word)
            
    def _add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.next_idx
            self.idx_to_word[self.next_idx] = word
            self.next_idx += 1
            
    def encode(self, text):
        """Кодирует текст задачи в вектор"""
        # Токенизация и приведение к нижнему регистру
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Вектор признаков
        vec = np.zeros(self.embedding_dim) # Инициализируем нулями
        
        # Простое мешковое представление (Bag-of-Words) с учётом частоты
        for word in words:
            if word in self.word_to_idx:
                vec[self.word_to_idx[word] % self.embedding_dim] += 1 # Используем хеширование для ограничения размера
            
        # Добавляем числовые параметры
        numbers = re.findall(r'\b\d+\b', text)
        for num_str in numbers:
            num = int(num_str)
            # Улучшенное кодирование чисел: можно использовать несколько признаков
            if 0 <= num < 1000: 
                vec[self.word_to_idx.get("число", 0) % self.embedding_dim] += 1 
                vec[self.word_to_idx.get("N", 1) % self.embedding_dim] = num / 100.0 # Нормализованное значение
            
        # Добавляем признаки наличия ключевых конструкций
        if re.search(r'\bfor\b', text, re.IGNORECASE): vec[self.word_to_idx.get("for", 2) % self.embedding_dim] = 1
        if re.search(r'\bwhile\b', text, re.IGNORECASE): vec[self.word_to_idx.get("while", 3) % self.embedding_dim] = 1
        if re.search(r'\bif\b', text, re.IGNORECASE): vec[self.word_to_idx.get("если", 4) % self.embedding_dim] = 1
        
        return vec


# ==================== УЛУЧШЕННЫЙ ИНТЕРФЕЙС ====================

class ImprovedNeuralInterface:
    """Интерфейс для взаимодействия с нейросетью"""
    
    def __init__(self, model_path="improved_neural_solver.pkl"):
        self.model_path = model_path
        self.generator = ImprovedTaskGenerator()
        self.encoder = ImprovedTaskEncoder()
        self.nn = None
        self._load_model()
        
    def _load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.nn = pickle.load(f)
            print("✅ Улучшенная модель успешно загружена!")
        except FileNotFoundError:
            print("⚠️ Файл модели не найден. Необходимо сначала обучить модель.")
            self.nn = ImprovedDeepNeuralSolver()
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            self.nn = ImprovedDeepNeuralSolver()
            
    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.nn, f)
        print(f"✅ Улучшенная модель сохранена в {self.model_path}")
        
    def train_model(self, epochs=100, batch_size=128, num_samples=20000):
        """Обучение модели"""
        print("Начинаем обучение улучшенной нейросети...")
        
        X, y = self.generate_dataset(num_samples)
        
        print(f"\nРазмер обучающего набора: {len(X)} примеров")
        print(f"Размер входных данных: {X.shape}")
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
                vec[i] = 1
                
        # Кодируем числовые параметры
        numbers = re.findall(r'\d+', solution)
        for i, num_str in enumerate(numbers):
            if i + len(constructions) < len(vec):
                vec[i + len(constructions)] = int(num_str) / 100.0 # Нормализация
                
        return vec

    def solve_task(self, task_text):
        """Решает задачу, поставленную на естественном языке"""
        print(f"\n▶️  Получена новая задача: '{task_text}'")
        
        # 1. Кодирование задачи
        task_vector = self.encoder.encode(task_text)
        
        # 2. Прямой проход через нейросеть
        predicted_solution_vector = self.nn.forward(task_vector.reshape(1, -1), training=False)
        
        # 3. Декодирование результата
        solution_code = self._process_output(predicted_solution_vector, task_text)
        
        print("\n🔡 Сгенерированный код:")
        print(solution_code)
        
        # 4. Выполнение кода
        print("\n▶️  Выполнение сгенерированного кода...")
        try:
            exec(solution_code, globals())
            print("\n✅ Код успешно выполнен!")
        except Exception as e:
            print(f"\n❌ Ошибка при выполнении кода: {e}")
            
        return solution_code

    def _process_output(self, output_vector, original_task):
        """Преобразует выход нейросети в исполняемый код"""
        output_vector = output_vector.flatten()
        
        # Определяем наиболее вероятный тип задачи
        task_type_map = {
            0: 'fixed_loop', 1: 'range_loop', 2: 'while_condition', 3: 'computation',
            4: 'char_pattern', 5: 'repeat_sentence', 6: 'star_rectangle',
            7: 'numbered_string_print', 8: 'squares_up_to_n', 9: 'star_triangle',
            10: 'population_growth', 11: 'sequence_until_keyword', 
            12: 'sequence_until_multiple_keywords', 13: 'sequence_divisible_by_7',
            14: 'sequence_sum_until_negative', 15: 'sequence_count_fives'
        }
        
        # Используем argmax для определения основного типа задачи
        main_task_idx = np.argmax(output_vector[:len(task_type_map)])
        task_type = task_type_map.get(main_task_idx, 'default')

        # Извлекаем числовые параметры из оригинальной задачи
        numbers = [int(n) for n in re.findall(r'\b\d+\b', original_task)]
        
        # Генерируем код на основе типа задачи и параметров
        code = "def solve():\n"
        
        if task_type == 'fixed_loop':
            count = numbers[0] if numbers else 10
            code += f"    for i in range({count}):\n        print('Hello World')\n"
        elif task_type == 'range_loop':
            start = numbers[0] if len(numbers) > 0 else 1
            end = numbers[1] if len(numbers) > 1 else 10
            code += f"    for i in range({start}, {end} + 1):\n        print(i)\n"
        elif task_type == 'while_condition':
            code += "    while True:\n        line = input()\n        if line == '0':\n            break\n        print(line)\n"
        elif task_type == 'computation':
            if 'сумм' in original_task:
                n = numbers[0] if numbers else 10
                code += f"    print(sum(range(1, {n} + 1)))\n"
            elif 'факториал' in original_task:
                n = numbers[0] if numbers else 5
                code += f"    f = 1\n    for i in range(1, {n} + 1):\n        f *= i\n    print(f)\n"
        elif task_type == 'char_pattern':
            code += "    print('AAA\nAAA\nAAA\nAAA\nAAA\nAAA\nBBBB\nBBBB\nBBBB\nBBBB\nBBBB\nE\nTTTTT\nTTTTT\nTTTTT\nTTTTT\nTTTTT\nTTTTT\nTTTTT\nTTTTT\nTTTTT\nG')\n"
        elif task_type == 'repeat_sentence':
            count = numbers[0] if numbers else 3
            code += f"    s = input()\n    for _ in range({count}):\n        print(s)\n"
        elif task_type == 'star_rectangle':
            height = numbers[0] if numbers else 5
            width = numbers[1] if len(numbers) > 1 else 19
            code += f"    for _ in range({height}):\n        print('*' * {width})\n"
        elif task_type == 'numbered_string_print':
            text = re.search(r'"(.*?)"', original_task)
            text = text.group(1) if text else "text"
            code += f"    for i in range(10):\n        print(f'{{i}} {text}')\n"
        elif task_type == 'squares_up_to_n':
            n = numbers[0] if numbers else 10
            code += f"    for i in range({n} + 1):\n        print(f'Квадрат числа {{i}} равен {{i*i}}')\n"
        elif task_type == 'star_triangle':
            height = numbers[0] if numbers else 5
            code += f"    for i in range(1, {height} + 1):\n        print('*' * i)\n"
        elif task_type == 'population_growth':
            start_pop = numbers[0] if len(numbers) > 0 else 1000
            rate = numbers[1] if len(numbers) > 1 else 10
            days = numbers[2] if len(numbers) > 2 else 5
            code += f"    pop = {start_pop}\n    for day in range(1, {days} + 1):\n        pop *= (1 + {rate}/100.0)\n        print(f'День {{day}}: {{int(pop)}}')\n"
        elif task_type == 'sequence_until_keyword':
            keyword = 'КОНЕЦ'
            if 'конец' in original_task: keyword = 'конец'
            code += f"    while True:\n        s = input()\n        if s == '{keyword}':\n            break\n        print(s)\n"
        elif task_type == 'sequence_until_multiple_keywords':
            code += "    count = 0\n    while True:\n        s = input()\n        if s in ['стоп', 'хватит', 'достаточно']:\n            break\n        count += 1\n    print(count)\n"
        elif task_type == 'sequence_divisible_by_7':
            code += "    while True:\n        try:\n            n = int(input())\n            if n % 7 == 0:\n                print(n)\n        except (ValueError, EOFError):\n            break\n"
        elif task_type == 'sequence_sum_until_negative':
            code += "    total = 0\n    while True:\n        n = int(input())\n        if n < 0:\n            break\n        total += n\n    print(total)\n"
        elif task_type == 'sequence_count_fives':
            code += "    count = 0\n    while True:\n        try:\n            grade = int(input())\n            if grade == 5:\n                count += 1\n        except (ValueError, EOFError):\n            break\n    print(count)\n"
        else:
            code += "    print('Не удалось распознать задачу.')\n"
            
        code += "\nsolve()"
        return code

# ======================== ГЛАВНЫЙ КОД ========================

if __name__ == '__main__':
    # Создание и обучение модели
    interface = ImprovedNeuralInterface()
    
    # Проверка, обучена ли модель
    if interface.nn.t == 0: # Простой флаг, что модель новая
        interface.train_model(epochs=50, num_samples=30000) # Увеличил количество эпох и сэмплов
        interface.save_model()

    # Примеры решения задач
    interface.solve_task("Напиши программу, которая выводит 10 строк, пронумерованных от 0 до 9 каждая, с указанной строкой текста: \"Hello, AI!\"")
    interface.solve_task("Напиши программу, которая выводит звёздный треугольник высотой 7")
    interface.solve_task("Напиши программу, которая предсказывает размер популяции организмов. Стартовое количество: 2000, среднесуточное увеличение: 15%, количество дней: 10.")
    interface.solve_task("Напиши программу, которая выводит члены данной последовательности до первого встретившегося слова \"стоп\"")
    interface.solve_task("Напиши программу, которая выводит сумму всех членов данной последовательности до первого отрицательного числа")
    interface.solve_task("Напиши программу, которая выводит количество пятерок в данной последовательности оценок (от 1 до 5)")

    # Интерактивный режим
    print("\n🤖 Нейросеть готова к работе! Введите вашу задачу:")
    while True:
        try:
            user_task = input("> ")
            if user_task.lower() in ['exit', 'выход']:
                break
            interface.solve_task(user_task)
        except KeyboardInterrupt:
            break
    print("\n👋 До свидания!")

    def _gen_fixed_loop_solution(self, count):
        return f"""def solve():
    for i in range({count}):
        print("Python is awesome!")

solve()"""
    
    def _gen_range_loop_solution(self, start, end, step):
        return f"""def solve():
    for i in range({start}, {end + 1}, {step}):
        print(i)

solve()"""
    
    def _gen_while_solution(self):
        return """def solve():
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

solve()"""
    
    def _gen_computation_solution(self, op):
        if op == "sum":
            return """def solve():
    n = int(input("Введите N: "))
    total = sum(range(1, n + 1))
    print(f"Сумма: {total}")

solve()"""
        elif op == "factorial":
            return """def solve():
    n = int(input("Введите число: "))
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    print(f"Факториал: {fact}")

solve()"""
        else:
            return self._gen_default_solution()

    def _gen_char_pattern_solution(self):
        return """def solve():
    for _ in range(6):
        print("AAA")
    for _ in range(5):
        print("BBBB")
    print("E")
    for _ in range(9):
        print("TTTTT")
    print("G")

solve()"""

    def _gen_repeat_sentence_solution(self):
        return """def solve():
    sentence = input()
    count = int(input())
    for _ in range(count):
        print(sentence)

solve()"""

    def _gen_star_rectangle_solution(self):
        return """def solve():
    n = int(input())
    for _ in range(n):
        print("*" * 19)

solve()"""

    def _gen_numbered_string_print_solution(self):
        return """def solve():
    text = input()
    for i in range(10):
        print(f"{i} {text}")

solve()"""

    def _gen_squares_up_to_n_solution(self):
        return """def solve():
    n = int(input())
    for i in range(n + 1):
        print(f"Квадрат числа {i} равен {i*i}")

solve()"""

    def _gen_star_triangle_solution(self):
        return """def solve():
    n = int(input())
    for i in range(n + 1):
        print("*" * i)

solve()"""

    def _gen_population_growth_solution(self):
        return """def solve():
    start_pop = int(input())
    growth_rate = float(input())
    days = int(input())

    current_pop = start_pop
    for day in range(days):
        current_pop *= (1 + growth_rate)
        print(f"День {day+1}: {int(current_pop)}")

solve()"""

    def _gen_sequence_until_keyword_solution(self, keyword):
        return f"""def solve():
    items = []
    while True:
        item = input()
        if item == \"{keyword}\":
            break
        items.append(item)
    for item in items:
        print(item)

solve()"""

    def _gen_sequence_until_multiple_keywords_solution(self, keywords):
        keywords_str = ", ".join([f'\"{k}\"' for k in keywords])
        return f"""def solve():
    count = 0
    while True:
        item = input()
        if item in [{keywords_str}]:
            break
        count += 1
    print(f"Количество членов: {{count}}")

solve()"""

    def _gen_sequence_divisible_by_7_solution(self):
        return """def solve():
    numbers = []
    while True:
        num_str = input()
        if not num_str:
            break
        num = int(num_str)
        if num % 7 == 0:
            numbers.append(num)
    for num in numbers:
        print(num)

solve()"""

    def _gen_sequence_sum_until_negative_solution(self):
        return """def solve():
    total_sum = 0
    while True:
        num = int(input())
        if num < 0:
            break
        total_sum += num
    print(f"Сумма: {{total_sum}}")

solve()"""

    def _gen_sequence_count_fives_solution(self):
        return """def solve():
    count_fives = 0
    while True:
        grade_str = input()
        if not grade_str:
            break
        grade = int(grade_str)
        if grade == 5:
            count_fives += 1
    print(f"Количество пятерок: {{count_fives}}")

solve()"""

    def _gen_default_solution(self):
        return """def solve():
    print("Hello, World!")

solve()"""

    def _gen_fixed_loop_solution(self, count):
        return f"""def solve():
    for i in range({count}):
        print("Python is awesome!")

solve()"""

