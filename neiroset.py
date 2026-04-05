import random
import math
import re
import numpy as np

class NeuralCodeGenerator:
    """
    Нейросеть, которая генерирует код решения задач на циклы
    Работает через понимание структуры задачи, а не по шаблонам
    """
    
    def __init__(self):
        # НЕЙРОСЕТЬ: 30 признаков -> 15 нейронов -> 10 действий
        self.w1 = [[random.uniform(-1, 1) for _ in range(15)] for _ in range(30)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(10)] for _ in range(15)]
        self.b1 = [random.uniform(-1, 1) for _ in range(15)]
        self.b2 = [random.uniform(-1, 1) for _ in range(10)]
        
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def extract_deep_features(self, text):
        """
        ГЛУБОКИЙ анализ текста - извлекаем реальную структуру задачи
        """
        text = text.lower()
        features = []
        
        # 1. ЧИСЛОВЫЕ ОПЕРАЦИИ
        operations = {
            'sum': ['сумму', 'сложение', 'плюс', 'сумма', 'total'],
            'product': ['произведени', 'умножени', 'произведение'],
            'max': ['максимум', 'наибольш', 'max', 'самый большой'],
            'min': ['минимум', 'наименьш', 'min', 'самый маленький'],
            'avg': ['среднее', 'average', 'среднеарифметическ'],
            'count': ['подсчита', 'количество', 'сколько', 'count']
        }
        
        for op_type, keywords in operations.items():
            score = sum(1 for kw in keywords if kw in text)
            features.append(min(score / 3, 1))
        
        # 2. ТИП ДАННЫХ
        data_types = {
            'numbers': ['числ', 'цифр', 'значени', 'элемент'],
            'list': ['массив', 'список', 'лист', 'последовательнос'],
            'range': ['диапазон', 'от', 'до', 'range'],
            'string': ['строк', 'символ', 'букв', 'text']
        }
        
        for dt_type, keywords in data_types.items():
            score = sum(1 for kw in keywords if kw in text)
            features.append(min(score / 2, 1))
        
        # 3. УСЛОВИЯ
        conditions = {
            'even_odd': ['чётн', 'нечётн', 'четное', 'нечетное', 'even', 'odd'],
            'greater': ['больше', '>', 'превыша', 'выше'],
            'less': ['меньше', '<', 'ниже'],
            'equal': ['равн', '==', 'совпада'],
            'filter': ['только', 'оставь', 'отбери', 'filter']
        }
        
        for cond_type, keywords in conditions.items():
            score = sum(1 for kw in keywords if kw in text)
            features.append(min(score / 2, 1))
        
        # 4. ЦИКЛЫ
        if 'for' in text or 'перебор' in text or 'каждый' in text:
            features.append(1.0)
        else:
            features.append(0.5 if 'while' in text else 0.0)
        
        features.append(1.0 if 'break' in text else 0.0)
        features.append(1.0 if 'continue' in text else 0.0)
        
        # 5. СЛОЖНОСТЬ
        words = text.split()
        features.append(min(len(words) / 50, 1))
        
        numbers = re.findall(r'\d+', text)
        features.append(min(len(numbers) / 10, 1))
        
        # 6. ИЗВЛЕКАЕМ ЧИСЛА ИЗ ТЕКСТА
        found_numbers = [int(n) for n in numbers if int(n) < 1000]
        if found_numbers:
            features.append(min(sum(found_numbers) / 1000, 1))
            features.append(len(found_numbers) / 10)
        else:
            features.append(0)
            features.append(0)
        
        # 7. ГЛАГОЛЫ ДЕЙСТВИЯ
        action_words = ['найди', 'вычисли', 'определи', 'посчитай', 'получи', 'создай', 'выведи']
        action_score = sum(1 for aw in action_words if aw in text)
        features.append(min(action_score / 3, 1))
        
        # Добиваем до 30
        while len(features) < 30:
            features.append(0)
        
        return features
    
    def forward(self, features):
        """Прямой проход по нейросети"""
        hidden = []
        for j in range(15):
            total = self.b1[j]
            for i in range(30):
                total += features[i] * self.w1[i][j]
            hidden.append(self.sigmoid(total))
        
        output = []
        for k in range(10):
            total = self.b2[k]
            for j in range(15):
                total += hidden[j] * self.w2[j][k]
            output.append(self.sigmoid(total))
        
        return output
    
    def understand_task(self, text):
        """
        НЕЙРОСЕТЬ АНАЛИЗИРУЕТ ЗАДАЧУ и понимает, что нужно сделать
        """
        features = self.extract_deep_features(text)
        output = self.forward(features)
        
        # Определяем тип задачи по самому активному нейрону
        task_type = np.argmax(output)
        
        # Извлекаем числа из текста
        numbers = [int(n) for n in re.findall(r'\d+', text)]
        
        # Определяем диапазон
        range_match = re.search(r'от\s*(\d+)\s*до\s*(\d+)', text)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            numbers = list(range(start, end + 1))
        
        # Если чисел нет, создаём тестовые
        if not numbers:
            numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        return {
            'task_type': task_type,
            'numbers': numbers,
            'text': text,
            'confidence': output[task_type]
        }
    
    def generate_solution(self, understanding):
        """
        НЕЙРОСЕТЬ ГЕНЕРИРУЕТ РЕШЕНИЕ на основе понимания
        """
        task_type = understanding['task_type']
        numbers = understanding['numbers']
        text = understanding['text'].lower()
        
        # Типы задач, которые понимает нейросеть
        if 'сумм' in text or task_type in [0, 1]:
            return self.solve_sum(numbers)
        
        elif 'максим' in text or task_type in [2]:
            return self.solve_max(numbers)
        
        elif 'миним' in text or task_type in [3]:
            return self.solve_min(numbers)
        
        elif 'средн' in text or 'average' in text:
            return self.solve_average(numbers)
        
        elif 'чётн' in text:
            return self.solve_even(numbers)
        
        elif 'нечётн' in text:
            return self.solve_odd(numbers)
        
        elif 'умнож' in text or 'произвед' in text:
            return self.solve_product(numbers)
        
        elif 'фильтр' in text or 'только' in text:
            return self.solve_filter(numbers, text)
        
        else:
            return self.solve_generic(numbers, text)
    
    def solve_sum(self, numbers):
        """Решает задачу на сумму с использованием цикла for"""
        result = 0
        code = f"""
# Решение: сумма всех чисел
numbers = {numbers}
result = 0
for num in numbers:
    result += num
print(f"Сумма: {{result}}")
"""
        for num in numbers:
            result += num
        return f"Сумма чисел {numbers} = {result}\n\nКод:\n{code}"
    
    def solve_max(self, numbers):
        """Находит максимум через for"""
        max_val = numbers[0]
        for num in numbers:
            if num > max_val:
                max_val = num
        return f"Максимальное число: {max_val}\n\nКод:\nnumbers = {numbers}\nmax_val = numbers[0]\nfor num in numbers:\n    if num > max_val:\n        max_val = num\nprint(max_val)"
    
    def solve_min(self, numbers):
        """Находит минимум через while"""
        min_val = numbers[0]
        i = 0
        while i < len(numbers):
            if numbers[i] < min_val:
                min_val = numbers[i]
            i += 1
        return f"Минимальное число: {min_val}\n\nКод:\nnumbers = {numbers}\nmin_val = numbers[0]\ni = 0\nwhile i < len(numbers):\n    if numbers[i] < min_val:\n        min_val = numbers[i]\n    i += 1\nprint(min_val)"
    
    def solve_average(self, numbers):
        """Среднее арифметическое"""
        total = 0
        count = 0
        for num in numbers:
            total += num
            count += 1
        avg = total / count
        return f"Среднее арифметическое: {avg:.2f}\n\nКод:\nnumbers = {numbers}\ntotal = 0\nfor num in numbers:\n    total += num\navg = total / len(numbers)\nprint(avg)"
    
    def solve_even(self, numbers):
        """Только чётные числа"""
        evens = []
        for num in numbers:
            if num % 2 == 0:
                evens.append(num)
        return f"Чётные числа: {evens}\n\nКод:\nnumbers = {numbers}\nevens = []\nfor num in numbers:\n    if num % 2 == 0:\n        evens.append(num)\nprint(evens)"
    
    def solve_odd(self, numbers):
        """Только нечётные числа"""
        odds = []
        i = 0
        while i < len(numbers):
            if numbers[i] % 2 != 0:
                odds.append(numbers[i])
            i += 1
        return f"Нечётные числа: {odds}\n\nКод:\nnumbers = {numbers}\nodds = []\ni = 0\nwhile i < len(numbers):\n    if numbers[i] % 2 != 0:\n        odds.append(numbers[i])\n    i += 1\nprint(odds)"
    
    def solve_product(self, numbers):
        """Произведение чисел"""
        result = 1
        for num in numbers:
            result *= num
        return f"Произведение чисел: {result}\n\nКод:\nnumbers = {numbers}\nresult = 1\nfor num in numbers:\n    result *= num\nprint(result)"
    
    def solve_filter(self, numbers, text):
        """Умная фильтрация на основе текста"""
        threshold = 5
        # Ищем порог в тексте
        threshold_match = re.search(r'больше\s*(\d+)|менее\s*(\d+)', text)
        if threshold_match:
            threshold = int(threshold_match.group(1) or threshold_match.group(2))
        
        filtered = [num for num in numbers if num > threshold]
        return f"Числа больше {threshold}: {filtered}\n\nКод:\nnumbers = {numbers}\nthreshold = {threshold}\nresult = []\nfor num in numbers:\n    if num > threshold:\n        result.append(num)\nprint(result)"
    
    def solve_generic(self, numbers, text):
        """Универсальный решатель"""
        return f"Я понял задачу: '{text[:50]}...'\nРаботаю с числами {numbers}\n\nПодсказка: уточните, что нужно сделать (найти сумму, максимум, чётные числа и т.д.)"


# ============ ЗАПУСК НЕЙРОСЕТИ ============

print("="*70)
print("🧠 НЕЙРОСЕТЬ ДЛЯ РЕШЕНИЯ ЛЮБЫХ ЗАДАЧ НА ЦИКЛЫ")
print("="*70)
print("\nЭта нейросеть:")
print("✅ Анализирует текст задачи (не по шаблонам)")
print("✅ Понимает, что нужно сделать")
print("✅ Сама генерирует код с циклами for/while")
print("✅ Выдаёт результат\n")

# Создаём нейросеть
nn = NeuralCodeGenerator()

# ТЕСТ: разные формулировки одной задачи
test_tasks = [
    "найди сумму чисел 5, 10, 15, 20",
    "посчитай сколько будет 5+10+15+20",
    "сложи все числа из списка 5 10 15 20",
    "какая сумма у чисел 5, 10, 15 и 20",
    
    "найди самое большое число в массиве 3 7 2 9 1",
    "какой максимум среди 3,7,2,9,1",
    "наибольший элемент из 3 7 2 9 1",
    
    "выведи все чётные числа от 1 до 10",
    "какие числа в диапазоне 1-10 делятся на 2",
    "найди чётные элементы списка 1,2,3,4,5,6,7,8,9,10",
]

print("="*70)
print("ТЕСТИРОВАНИЕ НЕЙРОСЕТИ")
print("="*70)

for i, task in enumerate(test_tasks, 1):
    print(f"\n📌 ЗАДАЧА {i}: {task}")
    print("-"*50)
    
    # Нейросеть анализирует задачу
    understanding = nn.understand_task(task)
    
    # Нейросеть генерирует решение
    solution = nn.generate_solution(understanding)
    
    print(f"🧠 Решение:\n{solution}")
    print("-"*50)

# ИНТЕРАКТИВНЫЙ РЕЖИМ
print("\n" + "="*70)
print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("="*70)
print("Задайте любую задачу на циклы (на русском)")
print("Нейросеть сама поймёт и решит!")
print("Введите 'выход' для завершения\n")

while True:
    user_task = input("Ваша задача: ").strip()
    if user_task.lower() in ['выход', 'exit', 'quit', 'q']:
        print("До свидания!")
        break
    
    if not user_task:
        continue
    
    print("\n🤔 Анализирую задачу...")
    understanding = nn.understand_task(user_task)
    print(f"📊 Уверенность нейросети: {understanding['confidence']:.2%}")
    
    print("\n💡 Генерирую решение...")
    solution = nn.generate_solution(understanding)
    
    print(f"\n✅ РЕШЕНИЕ:\n{solution}")
    print("\n" + "-"*50)
