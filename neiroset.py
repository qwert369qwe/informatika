import numpy as np
import random
import re
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==================== СПИСОК ТИПОВ ЗАДАЧ ====================
TASK_TYPES = [
    'fixed_loop', 'range_loop', 'while_condition', 'computation',
    'char_pattern', 'repeat_sentence', 'star_rectangle',
    'numbered_string_print', 'squares_up_to_n', 'star_triangle',
    'population_growth', 'sequence_until_keyword',
    'sequence_until_multiple_keywords', 'sequence_divisible_by_7',
    'sequence_sum_until_negative', 'sequence_count_fives'
]
NUM_TYPES = len(TASK_TYPES)
NUM_PARAMS = 4   # count, start, end, step

# ==================== УЛУЧШЕННАЯ НЕЙРОСЕТЬ (ПРЕДСКАЗЫВАЕТ ТИП + ПАРАМЕТРЫ) ====================
class ImprovedDeepNeuralSolver:
    def __init__(self, input_dim=256, hidden_dims=[512, 256, 128], output_dim=NUM_TYPES + NUM_PARAMS):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W1 = np.random.randn(input_dim, hidden_dims[0]) * np.sqrt(2.0 / input_dim)
        self.W2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * np.sqrt(2.0 / hidden_dims[0])
        self.W3 = np.random.randn(hidden_dims[1], hidden_dims[2]) * np.sqrt(2.0 / hidden_dims[1])
        self.W4 = np.random.randn(hidden_dims[2], output_dim) * np.sqrt(2.0 / hidden_dims[2])

        self.b1 = np.zeros(hidden_dims[0])
        self.b2 = np.zeros(hidden_dims[1])
        self.b3 = np.zeros(hidden_dims[2])
        self.b4 = np.zeros(output_dim)

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
        self.learning_rate = 0.0005

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def forward(self, x, training=True):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        if training:
            self.dropout1 = np.random.binomial(1, 1-self.dropout_rate, size=self.a1.shape) / (1-self.dropout_rate)
            self.a1 *= self.dropout1

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        if training:
            self.dropout2 = np.random.binomial(1, 1-self.dropout_rate, size=self.a2.shape) / (1-self.dropout_rate)
            self.a2 *= self.dropout2

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.relu(self.z3)

        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.type_logits = self.z4[:, :NUM_TYPES]
        self.params_logits = self.z4[:, NUM_TYPES:]
        self.type_probs = self.softmax(self.type_logits)
        self.params_pred = self.sigmoid(self.params_logits)
        return self.type_probs, self.params_pred

    def compute_loss(self, y_type_true, y_params_true, y_type_pred, y_params_pred):
        ce = -np.mean(np.sum(y_type_true * np.log(y_type_pred + 1e-8), axis=1))
        mse = np.mean((y_params_true - y_params_pred) ** 2)
        return ce + mse

    def backward(self, x, y_type_true, y_params_true, y_type_pred, y_params_pred):
        self.t += 1
        d_type = y_type_pred - y_type_true
        d_params = (y_params_pred - y_params_true) * y_params_pred * (1 - y_params_pred)
        d_output = np.concatenate([d_type, d_params], axis=1)

        d_W4 = np.dot(self.a3.T, d_output)
        d_b4 = np.sum(d_output, axis=0)
        d_a3 = np.dot(d_output, self.W4.T)

        d_z3 = d_a3 * self.relu_derivative(self.z3)
        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0)
        d_a2 = np.dot(d_z3, self.W3.T)

        d_z2 = d_a2 * self.relu_derivative(self.z2)
        if hasattr(self, 'dropout2'): d_z2 *= self.dropout2
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)
        d_a1 = np.dot(d_z2, self.W2.T)

        d_z1 = d_a1 * self.relu_derivative(self.z1)
        if hasattr(self, 'dropout1'): d_z1 *= self.dropout1
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

        return self.compute_loss(y_type_true, y_params_true, y_type_pred, y_params_pred)

    def _adam_update(self, param, grad, m, v):
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# ==================== УЛУЧШЕННЫЙ ГЕНЕРАТОР (СОХРАНЁН ВЕСЬ СТАРЫЙ ФУНКЦИОНАЛ) ====================
class ImprovedTaskGenerator:
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
        # 3. Циклы while
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
        # 5. Символьные паттерны
        templates.append({'pattern': 'печати следующей последовательности символов', 'type': 'char_pattern'})
        # 6. Повторение предложения
        templates.append({'pattern': 'повторяет данное предложение нужное количество раз', 'type': 'repeat_sentence'})
        # 7. Звёздный прямоугольник
        templates.append({'pattern': 'печатает звёздный прямоугольник', 'type': 'star_rectangle'})
        # 8. Нумерованные строки
        templates.append({'pattern': 'выводит 10 строк, пронумерованных от 0 до 9 каждая, с указанной строкой текста', 'type': 'numbered_string_print'})
        # 9. Квадраты чисел
        templates.append({'pattern': 'квадрат числа <текущее число> равен <квадрат текущего числа>', 'type': 'squares_up_to_n'})
        # 10. Звёздный треугольник
        templates.append({'pattern': 'выводит звёздный треугольник', 'type': 'star_triangle'})
        # 11. Популяция
        templates.append({'pattern': 'предсказывает размер популяции организмов', 'type': 'population_growth'})
        # 12. Последовательность до ключевого слова
        templates.append({'pattern': 'выводит члены данной последовательности до слова "КОНЕЦ"', 'type': 'sequence_until_keyword', 'keyword': 'КОНЕЦ'})
        templates.append({'pattern': 'выводит члены данной последовательности до "КОНЕЦ" или "конец"', 'type': 'sequence_until_keyword', 'keyword': 'КОНЕЦ'})
        templates.append({'pattern': 'выводит общее количество членов данной последовательности', 'type': 'sequence_until_multiple_keywords', 'keywords': ['стоп', 'хватит', 'достаточно']})
        # 13. Делящиеся на 7
        templates.append({'pattern': 'выводит члены данной последовательности, делящиеся на 7', 'type': 'sequence_divisible_by_7'})
        # 14. Сумма до отрицательного
        templates.append({'pattern': 'выводит сумму всех членов данной последовательности до отрицательного числа', 'type': 'sequence_sum_until_negative'})
        # 15. Количество пятёрок
        templates.append({'pattern': 'выводит количество пятерок в последовательности оценок', 'type': 'sequence_count_fives'})
        return templates

    def generate_task(self):
        template = random.choice(self.templates)
        prefixes = [
            "Напишите программу, которая",
            "Требуется написать код, который",
            "Создайте программу для",
            "Реализуйте алгоритм, который"
        ]
        suffix = random.choice(prefixes)
        task_text = f'{suffix} {template["pattern"]}'
        return task_text, template

    # ---- ВСЕ ОРИГИНАЛЬНЫЕ МЕТОДЫ ГЕНЕРАЦИИ РЕШЕНИЙ (СОХРАНЕНЫ БЕЗ ИЗМЕНЕНИЙ) ----
    def generate_solution(self, template):
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

    # ---- НОВЫЙ МЕТОД ДЛЯ ПОЛУЧЕНИЯ ТИПА И ПАРАМЕТРОВ (НЕ УДАЛЯЕТ СТАРЫЙ) ----
    def generate_solution_and_target(self, template):
        """Возвращает (код, тип_задачи, словарь_параметров)"""
        task_type = template["type"]
        # Извлекаем параметры в универсальный словарь
        params = {}
        if task_type == "fixed_loop":
            params['count'] = template.get('count', 10)
        elif task_type == "range_loop":
            params['start'] = template.get('start', 1)
            params['end'] = template.get('end', 10)
            params['step'] = template.get('step', 1)
        elif task_type == "computation":
            params['op'] = template.get('op', 'sum')
        elif task_type == "sequence_until_keyword":
            params['keyword'] = template.get('keyword', 'КОНЕЦ')
        elif task_type == "sequence_until_multiple_keywords":
            params['keywords'] = template.get('keywords', ['стоп', 'хватит', 'достаточно'])
        # Остальные типы не требуют числовых параметров, оставляем пустой словарь
        code = self.generate_solution(template)
        return code, task_type, params

# ==================== КОДИРОВЩИК ЗАДАЧ (БЕЗ ИЗМЕНЕНИЙ) ====================
class ImprovedTaskEncoder:
    def __init__(self, vocab_size=10000, embedding_dim=256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.next_idx = 0
        self._build_vocab_from_keywords()

    def _build_vocab_from_keywords(self):
        keywords = [
            "программа", "написать", "код", "реализовать", "алгоритм",
            "вывести", "напечатать", "повторить", "сделать", "итераций",
            "числа", "текст", "строки", "символы", "последовательность",
            "ввод", "вывод", "пока", "до", "если", "равен", "больше", "меньше",
            "сумма", "факториал", "среднее", "количество", "размер",
            "for", "while", "цикл", "диапазон", "от", "до", "шаг",
            "условие", "встретится", "пустой", "отрицательный", "конец",
            "стоп", "хватит", "достаточно", "делится", "неположительный",
            "раз", "число", "N", "M", "P", "дней", "высота", "ширина",
            "звездный", "прямоугольник", "треугольник", "квадрат", "оценка",
            "пятерок", "организм", "популяция", "процент", "среднесуточное",
            "AAA", "BBBB", "E", "TTTTT", "G", "строка", "предложение",
            "повторений", "высота", "ширина", "пронумерованных", "квадрат", "катет"
        ]
        for w in keywords:
            self._add_word(w)

    def _add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.next_idx
            self.idx_to_word[self.next_idx] = word
            self.next_idx += 1

    def encode(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        vec = np.zeros(self.embedding_dim)
        for w in words:
            if w in self.word_to_idx:
                vec[self.word_to_idx[w] % self.embedding_dim] += 1
        numbers = re.findall(r'\b\d+\b', text)
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num < 1000:
                vec[self.word_to_idx.get("число", 0) % self.embedding_dim] += 1
                vec[self.word_to_idx.get("N", 1) % self.embedding_dim] = num / 100.0
        if re.search(r'\bfor\b', text, re.IGNORECASE): vec[self.word_to_idx.get("for", 2) % self.embedding_dim] = 1
        if re.search(r'\bwhile\b', text, re.IGNORECASE): vec[self.word_to_idx.get("while", 3) % self.embedding_dim] = 1
        if re.search(r'\bif\b', text, re.IGNORECASE): vec[self.word_to_idx.get("если", 4) % self.embedding_dim] = 1
        return vec

# ==================== УЛУЧШЕННЫЙ ИНТЕРФЕЙС (С ОБУЧЕНИЕМ НА ТИПАХ) ====================
class ImprovedNeuralInterface:
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

    def generate_dataset(self, num_samples=30000):
        print(f"Генерация {num_samples} улучшенных задач...")
        X, Y_types, Y_params = [], [], []
        for i in range(num_samples):
            task_text, template = self.generator.generate_task()
            solution, task_type, params = self.generator.generate_solution_and_target(template)
            vec = self.encoder.encode(task_text)
            X.append(vec)
            # one-hot тип
            type_onehot = np.zeros(NUM_TYPES)
            if task_type in TASK_TYPES:
                type_onehot[TASK_TYPES.index(task_type)] = 1.0
            Y_types.append(type_onehot)
            # числовые параметры (нормализованные)
            param_vals = self._extract_numeric_params(params)
            Y_params.append(param_vals)
            if (i+1) % 5000 == 0:
                print(f"  Сгенерировано {i+1}/{num_samples} задач")
        return np.array(X), np.array(Y_types), np.array(Y_params)

    def _extract_numeric_params(self, params):
        """Извлекает до NUM_PARAMS числовых значений и нормализует в [0,1]"""
        vals = np.zeros(NUM_PARAMS)
        if 'count' in params:
            vals[0] = min(1.0, params['count'] / 200.0)
        if 'start' in params:
            vals[1] = min(1.0, params['start'] / 100.0)
        if 'end' in params:
            vals[2] = min(1.0, params['end'] / 100.0)
        if 'step' in params:
            vals[3] = min(1.0, params['step'] / 20.0)
        return vals

    def train_model(self, epochs=80, batch_size=128, num_samples=30000):
        print("Начинаем обучение улучшенной нейросети (предсказание типа + параметров)...")
        X, Y_types, Y_params = self.generate_dataset(num_samples)
        print(f"Размер обучающего набора: {len(X)} примеров")
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X, Y_types, Y_params = X[idx], Y_types[idx], Y_params[idx]
            total_loss = 0
            num_batches = 0
            for i in range(0, len(X), batch_size):
                bx = X[i:i+batch_size]
                by_type = Y_types[i:i+batch_size]
                by_param = Y_params[i:i+batch_size]
                pred_type, pred_param = self.nn.forward(bx)
                loss = self.nn.backward(bx, by_type, by_param, pred_type, pred_param)
                total_loss += loss
                num_batches += 1
            if epoch % 10 == 0:
                print(f"Эпоха {epoch}/{epochs}, Потери: {total_loss/num_batches:.6f}")
        print("\n✅ Обучение улучшенной нейросети завершено!")

    def solve_task(self, task_text):
        print(f"\n▶️  Получена новая задача: '{task_text}'")
        vec = self.encoder.encode(task_text).reshape(1, -1)
        pred_type, pred_param = self.nn.forward(vec, training=False)
        type_idx = np.argmax(pred_type[0])
        task_type = TASK_TYPES[type_idx]
        # Денормализуем параметры
        params = self._denormalize_params(pred_param[0])
        # Извлекаем также числа из исходного текста (для точности)
        numbers = [int(n) for n in re.findall(r'\b\d+\b', task_text)]
        # Генерируем код на основе предсказанного типа и параметров
        solution_code = self._generate_code_from_type(task_type, params, numbers, task_text)
        print("\n🔡 Сгенерированный код:")
        print(solution_code)
        print("\n▶️  Выполнение сгенерированного кода...")
        try:
            exec(solution_code, globals())
            print("\n✅ Код успешно выполнен!")
        except Exception as e:
            print(f"\n❌ Ошибка при выполнении кода: {e}")
        return solution_code

    def _denormalize_params(self, norm_params):
        p = {}
        p['count'] = int(norm_params[0] * 200)
        p['start'] = int(norm_params[1] * 100)
        p['end'] = int(norm_params[2] * 100)
        p['step'] = max(1, int(norm_params[3] * 20))
        return p

    def _generate_code_from_type(self, task_type, params, numbers, original_task):
        """Использует сохранённые методы генерации из ImprovedTaskGenerator"""
        # Создаём фейковый шаблон, чтобы вызвать нужный метод генератора
        template = {'type': task_type}
        if task_type == 'fixed_loop':
            count = numbers[0] if numbers else params.get('count', 10)
            template['count'] = count
        elif task_type == 'range_loop':
            template['start'] = numbers[0] if numbers else params.get('start', 1)
            template['end'] = numbers[1] if len(numbers) > 1 else params.get('end', 10)
            template['step'] = params.get('step', 1)
        elif task_type == 'computation':
            if 'сумм' in original_task:
                template['op'] = 'sum'
            elif 'факториал' in original_task:
                template['op'] = 'factorial'
            else:
                template['op'] = 'sum'
        elif task_type == 'sequence_until_keyword':
            if 'конец' in original_task.lower():
                template['keyword'] = 'конец'
            else:
                template['keyword'] = 'КОНЕЦ'
        elif task_type == 'sequence_until_multiple_keywords':
            template['keywords'] = ['стоп', 'хватит', 'достаточно']
        # Для остальных типов параметры не нужны
        return self.generator.generate_solution(template)

# ==================== ГЛАВНЫЙ КОД ====================
if __name__ == '__main__':
    interface = ImprovedNeuralInterface()
    if interface.nn.t == 0:
        interface.train_model(epochs=80, num_samples=40000)
        interface.save_model()

    # Тестируем на проблемной задаче
    interface.solve_task("Напишите программу, которая выводит текст «Python is awesome!» (без кавычек) 10 раз")
    interface.solve_task("Напиши программу, которая выводит звёздный треугольник высотой 7")
    interface.solve_task("Напиши программу, которая предсказывает размер популяции организмов. Стартовое количество: 2000, среднесуточное увеличение: 15%, количество дней: 10.")

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
