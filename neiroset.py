import random
import math
import re

class CycleTaskSolver:
    def __init__(self):
        # Нейросеть: 20 признаков задачи -> 10 нейронов -> 5 действий
        self.w1 = [[random.uniform(-1, 1) for _ in range(10)] for _ in range(20)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(5)] for _ in range(10)]
        self.b1 = [random.uniform(-1, 1) for _ in range(10)]
        self.b2 = [random.uniform(-1, 1) for _ in range(5)]
        
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def parse_task(self, task_text):
        """Превращаем текст задачи в числовые признаки"""
        features = []
        
        # Признаки: наличие ключевых слов
        keywords = ['for', 'while', 'range', 'сумму', 'произведение', 
                   'массив', 'список', 'цикл', 'каждый', 'элемент',
                   'вывести', 'найти', 'подсчитать', 'среднее', 'максимум',
                   'минимум', 'условие', 'if', 'break', 'continue']
        
        for kw in keywords:
            features.append(1 if kw in task_text.lower() else 0)
        
        # Числовые признаки
        numbers = re.findall(r'\d+', task_text)
        features.append(len(numbers) / 10)  # количество чисел
        features.append(len(task_text) / 200)  # длина задачи
        
        # Добиваем до 20 признаков
        while len(features) < 20:
            features.append(0)
            
        return features
    
    def forward(self, features):
        """Прямой проход по нейросети"""
        # Скрытый слой
        hidden = []
        for j in range(10):
            total = self.b1[j]
            for i in range(20):
                total += features[i] * self.w1[i][j]
            hidden.append(self.sigmoid(total))
        
        # Выходной слой (5 действий)
        output = []
        for k in range(5):
            total = self.b2[k]
            for j in range(10):
                total += hidden[j] * self.w2[j][k]
            output.append(self.sigmoid(total))
            
        return output
    
    def generate_solution(self, task_text):
        """Генерируем решение задачи на основе выхода нейросети"""
        features = self.parse_task(task_text)
        output = self.forward(features)
        
        # Определяем тип задачи по выходу нейросети
        task_type = output.index(max(output))
        
        solutions = {
            0: self.solve_sum_task,
            1: self.solve_count_task,
            2: self.solve_max_min_task,
            3: self.solve_pattern_task,
            4: self.solve_filter_task
        }
        
        return solutions.get(task_type, self.solve_generic)(task_text)
    
    def solve_sum_task(self, task_text):
        """Решает задачи на суммирование"""
        # Извлекаем числа из задачи
        numbers = re.findall(r'\d+', task_text)
        if numbers:
            nums = [int(n) for n in numbers]
            
            # РЕШАЕМ ЧЕРЕЗ ЦИКЛ FOR
            total = 0
            for num in nums:
                total += num
            
            # ИЛИ через while (для демонстрации)
            # i = 0
            # total = 0
            # while i < len(nums):
            #     total += nums[i]
            #     i += 1
            
            return f"Решение: сумма чисел {nums} = {total}\nКод:\nresult = 0\nfor num in {nums}:\n    result += num\nprint(result)  # {total}"
        return "Решение: нужны числа для суммирования"
    
    def solve_count_task(self, task_text):
        """Решает задачи на подсчёт"""
        numbers = re.findall(r'\d+', task_text)
        if 'чёт' in task_text.lower() or 'нечёт' in task_text.lower():
            nums = [int(n) for n in numbers] if numbers else [1,2,3,4,5,6,7,8,9,10]
            
            # РЕШАЕМ ЧЕРЕЗ ЦИКЛ WHILE
            count = 0
            i = 0
            while i < len(nums):
                if nums[i] % 2 == 0:  # чётные
                    count += 1
                i += 1
            
            return f"Решение: количество чётных чисел = {count}\nКод:\nnumbers = {nums}\ncount = 0\ni = 0\nwhile i < len(numbers):\n    if numbers[i] % 2 == 0:\n        count += 1\n    i += 1\nprint(count)  # {count}"
        return "Решение: задача на подсчёт (нужно уточнить условие)"
    
    def solve_max_min_task(self, task_text):
        """Решает задачи на поиск максимума/минимума"""
        numbers = re.findall(r'\d+', task_text)
        if numbers:
            nums = [int(n) for n in numbers]
            
            # РЕШАЕМ ЧЕРЕЗ ЦИКЛ FOR
            if 'максим' in task_text.lower():
                max_val = nums[0]
                for num in nums:
                    if num > max_val:
                        max_val = num
                return f"Решение: максимум = {max_val}\nКод:\nnumbers = {nums}\nmax_val = numbers[0]\nfor num in numbers:\n    if num > max_val:\n        max_val = num\nprint(max_val)  # {max_val}"
            else:
                min_val = nums[0]
                for num in nums:
                    if num < min_val:
                        min_val = num
                return f"Решение: минимум = {min_val}\nКод:\nnumbers = {nums}\nmin_val = numbers[0]\nfor num in numbers:\n    if num < min_val:\n        min_val = num\nprint(min_val)  # {min_val}"
        return "Решение: нужен список чисел"
    
    def solve_pattern_task(self, task_text):
        """Решает задачи на вывод узоров/паттернов"""
        numbers = re.findall(r'\d+', task_text)
        n = int(numbers[0]) if numbers else 5
        
        # РЕШАЕМ ЧЕРЕЗ ВЛОЖЕННЫЕ ЦИКЛЫ
        pattern = []
        for i in range(1, n + 1):
            row = []
            for j in range(1, i + 1):
                row.append(str(j))
            pattern.append(' '.join(row))
        
        result = '\n'.join(pattern)
        return f"Решение: треугольник из чисел\nКод:\nn = {n}\nfor i in range(1, n+1):\n    for j in range(1, i+1):\n        print(j, end=' ')\n    print()\n\nРезультат:\n{result}"
    
    def solve_filter_task(self, task_text):
        """Решает задачи на фильтрацию"""
        numbers = re.findall(r'\d+', task_text)
        if numbers:
            nums = [int(n) for n in numbers]
            
            # РЕШАЕМ ЧЕРЕЗ ЦИКЛ WHILE С УСЛОВИЕМ
            filtered = []
            i = 0
            while i < len(nums):
                if nums[i] > 5:  # фильтруем числа больше 5
                    filtered.append(nums[i])
                i += 1
            
            return f"Решение: числа > 5 из {nums} = {filtered}\nКод:\nnumbers = {nums}\nresult = []\ni = 0\nwhile i < len(numbers):\n    if numbers[i] > 5:\n        result.append(numbers[i])\n    i += 1\nprint(result)  # {filtered}"
        return "Решение: нужен список чисел для фильтрации"
    
    def solve_generic(self, task_text):
        """Универсальное решение для простых задач"""
        numbers = re.findall(r'\d+', task_text)
        if numbers:
            nums = [int(n) for n in numbers]
            
            # Базовый цикл for
            result = []
            for num in nums:
                result.append(num * 2)
            
            return f"Решение: каждый элемент умножен на 2\nКод:\nnumbers = {nums}\nresult = []\nfor num in numbers:\n    result.append(num * 2)\nprint(result)  # {result}"
        return "Решение: пример цикла\nКод:\nfor i in range(5):\n    print(f'Итерация {i}')"

# ============ ИСПОЛЬЗОВАНИЕ ============

# Создаём нейросеть-решатель задач
solver = CycleTaskSolver()

# Задачи для решения
tasks = [
    "Найди сумму чисел 5, 10, 15, 20",
    "Подсчитай количество чётных чисел в списке 1 2 3 4 5 6",
    "Найди максимальное число в массиве 3 7 2 9 1",
    "Выведи треугольник из чисел до 4",
    "Оставь только числа больше 5 из 2 8 3 9 1 7"
]

print("=" * 60)
print("НЕЙРОСЕТЬ РЕШАЕТ ЗАДАЧИ НА ЦИКЛЫ")
print("=" * 60)

for task in tasks:
    print(f"\n📌 ЗАДАЧА: {task}")
    print("-" * 40)
    solution = solver.generate_solution(task)
    print(solution)
    print("-" * 40)

# Интерактивный режим
print("\n\n🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("Введите свою задачу на циклы (или 'выход' для завершения):")

while True:
    user_task = input("\nВаша задача: ")
    if user_task.lower() in ['выход', 'exit', 'quit']:
        print("До свидания!")
        break
    
    solution = solver.generate_solution(user_task)
    print("\n🤖 НЕЙРОСЕТЬ РЕШИЛА:")
    print(solution)
