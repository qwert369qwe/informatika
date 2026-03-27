# ultimate_cycle_bot.py - МЕГА-РЕШАТОР ЗАДАЧ ПО ЦИКЛАМ
# Версия 5.1 - Исправленная и дополненная версия
# Решает ЛЮБЫЕ задачи: for, while, вложенные, алгоритмы, математика, строки, списки

import re
import math
import random
import sys
from typing import Dict, List, Tuple, Any, Optional

class UltimateCycleBot:
    def __init__(self):
        self.version = "5.1"
        self.solved_count = 0
        self.user_level = 1
        self.history = []

        # База знаний для анализа
        self.patterns = self._build_patterns()
        self.templates = self._build_templates()
        self.algorithms = self._build_algorithms()

    def _build_patterns(self) -> Dict:
        """Строит паттерны для распознавания задач"""
        return {
            # Базовые операции
            "print_numbers": [
                r"выведи\s+числа?\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"вывести\s+числа?\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"покажи\s+числа?\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
            ],
            "print_range": [
                r"выведи\s+числа?\s+от\s+(\d+)\s+до\s+(\d+)",
                r"числа\s+от\s+(\d+)\s+до\s+(\d+)",
            ],
            "sum_range": [
                r"сумм[уа]\s+чисел\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"найди\s+сумм[уа]\s+чисел\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"сложи\s+числа\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
            ],
            "product_range": [
                r"произведени[ея]\s+чисел\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"умножь\s+числа\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
            ],
            "even_numbers": [
                r"четн[ыу][ех]?\s+числа?\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"выведи\s+четн[ыу][ех]?\s+числа",
            ],
            "odd_numbers": [
                r"нечетн[ыу][ех]?\s+числа?\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"выведи\s+нечетн[ыу][ех]?\s+числа",
            ],
            "multiples": [
                r"кратн[ыу][ех]?\s+(\d+)\s+числа?\s+(от\s+)?(\d+)\s+(до|по)\s+(\d+)",
                r"числа[,\s]+кратные\s+(\d+)",
            ],

            # Алгоритмические задачи
            "factorial": [
                r"факториал\s+числа?\s+(\d+)",
                r"найди\s+факториал",
            ],
            "fibonacci": [
                r"фибоначч[иа]\s+(\d+)",
                r"числа\s+фибоначчи",
                r"fibonacci",
            ],
            "prime_numbers": [
                r"прост[ыу][ех]?\s+числа?\s+(до\s+)?(\d+)",
                r"простые\s+числа",
                r"prime\s+numbers",
            ],
            "perfect_numbers": [
                r"совершенн[ыу][ех]?\s+числа?\s+(до\s+)?(\d+)",
                r"perfect\s+numbers",
            ],
            "armstrong": [
                r"армстронг[а]?\s+числа?\s+(до\s+)?(\d+)",
                r"armstrong\s+numbers",
            ],
            "palindrome": [
                r"палиндром[ы]?\s+(до\s+)?(\d+)",
                r"palindrome\s+numbers",
            ],

            # Списки и массивы
            "list_find": [
                r"найди\s+(\d+)\s+в\s+списк[еа]\s+\[(.*?)\]",
                r"поиск\s+(\d+)\s+в\s+списк[еа]",
            ],
            "list_max": [
                r"максимальн[ыу][ех]?\s+число\s+в\s+списк[еа]\s+\[(.*?)\]",
                r"наибольше[ее]\s+число",
            ],
            "list_min": [
                r"минимальн[ыу][ех]?\s+число\s+в\s+списк[еа]\s+\[(.*?)\]",
                r"наименьше[ее]\s+число",
            ],
            "list_sum": [
                r"сумм[уа]\s+элементов\s+списка\s+\[(.*?)\]",
                r"сумма\s+списка",
            ],
            "list_average": [
                r"средне[ее]\s+арифметическ[ое]+\s+списка\s+\[(.*?)\]",
                r"среднее\s+списка",
            ],
            "list_sort": [
                r"сортир[уо][йв]?\s+список",
                r"отсортируй\s+список",
                r"sort\s+list",
            ],
            "list_reverse": [
                r"переверн[иу]?\s+список",
                r"обратн[ыу][йе]?\s+порядок",
            ],

            # Матрицы и вложенные циклы
            "matrix_create": [
                r"создай\s+матриц[уа]\s+(\d+)х(\d+)",
                r"таблица\s+(\d+)\s*[хx]\s*(\d+)",
            ],
            "multiplication_table": [
                r"таблиц[ау]\s+умножени[я]\s+(на\s+)?(\d+)",
                r"таблица\s+пифагора",
            ],
            "pyramid": [
                r"пирамид[куа]\s+(из\s+)?\*?\s+(высот[аой]\s+)?(\d+)",
                r"треугольник\s+из\s+звездочек",
            ],
            "diamond": [
                r"ромб\s+(высот[аой]\s+)?(\d+)",
                r"diamond",
            ],

            # Сложные алгоритмы
            "gcd": [
                r"нод\s+чисел?\s+(\d+)\s+и\s+(\d+)",
                r"наибольший\s+общий\s+делитель",
            ],
            "lcm": [
                r"нок\s+чисел?\s+(\d+)\s+и\s+(\d+)",
                r"наименьшее\s+общее\s+кратное",
            ],
            "binary_search": [
                r"двоичн[ыу][йе]?\s+поиск",
                r"binary\s+search",
            ],
            "bubble_sort": [
                r"пузырьк[оа][в]?[ая]?\s+сортировк[ау]",
                r"bubble\s+sort",
            ],
            "quick_sort": [
                r"быстра[я]\s+сортировк[ау]",
                r"quick\s+sort",
            ],
            "merge_sort": [
                r"сортировк[ау]\s+слияни[еем]",
                r"merge\s+sort",
            ],

            # Работа со строками
            "string_reverse": [
                r"переверн[иу]?\s+строк[уа]",
                r"обратн[ыу][йе]?\s+порядок\s+строки",
            ],
            "string_count": [
                r"подсчитай\s+символ[ы]?",
                r"сколько\s+символов",
            ],
            "string_palindrome": [
                r"провер[и]?\s+строк[уа]\s+на\s+палиндром",
                r"is\s+palindrome",
            ],
            "caesar_cipher": [
                r"шифр\s+цезар[я]",
                r"caesar\s+cipher",
            ],

            # Математические ряды
            "arithmetic_progression": [
                r"арифметическ[ая][я]\s+прогресси[яю]",
                r"арифм\.\s+прогрессия",
            ],
            "geometric_progression": [
                r"геометрическ[ая][я]\s+прогресси[яю]",
                r"геом\.\s+прогрессия",
            ],

            # Циклы while
            "while_until": [
                r"пока\s+(\w+)\s+(\<|\>|\<=|\>=|\==)\s+(\d+)",
                r"цикл\s+while",
            ],
            "infinite_loop": [
                r"бесконечн[ыу][йе]\s+цикл",
                r"while\s+true",
            ],

            # По умолчанию
            "unknown": [],
        }

    def _build_templates(self) -> Dict:
        """Шаблоны генерации кода"""
        return {
            "print_numbers": self._gen_print_numbers,
            "print_range": self._gen_print_range,
            "sum_range": self._gen_sum_range,
            "product_range": self._gen_product_range,
            "even_numbers": self._gen_even_numbers,
            "odd_numbers": self._gen_odd_numbers,
            "multiples": self._gen_multiples,
            "factorial": self._gen_factorial,
            "fibonacci": self._gen_fibonacci,
            "prime_numbers": self._gen_prime_numbers,
            "perfect_numbers": self._gen_perfect_numbers,
            "armstrong": self._gen_armstrong,
            "palindrome": self._gen_palindrome,
            "list_find": self._gen_list_find,
            "list_max": self._gen_list_max,
            "list_min": self._gen_list_min,
            "list_sum": self._gen_list_sum,
            "list_average": self._gen_list_average,
            "list_sort": self._gen_list_sort,
            "list_reverse": self._gen_list_reverse,
            "matrix_create": self._gen_matrix_create,
            "multiplication_table": self._gen_multiplication_table,
            "pyramid": self._gen_pyramid,
            "diamond": self._gen_diamond,
            "gcd": self._gen_gcd,
            "lcm": self._gen_lcm,
            "binary_search": self._gen_binary_search,
            "bubble_sort": self._gen_bubble_sort,
            "quick_sort": self._gen_quick_sort,
            "merge_sort": self._gen_merge_sort,
            "string_reverse": self._gen_string_reverse,
            "string_count": self._gen_string_count,
            "string_palindrome": self._gen_string_palindrome,
            "caesar_cipher": self._gen_caesar_cipher,
            "arithmetic_progression": self._gen_arithmetic_progression,
            "geometric_progression": self._gen_geometric_progression,
            "while_until": self._gen_while_until,
            "infinite_loop": self._gen_infinite_loop,
        }

    def _build_algorithms(self) -> Dict:
        """Сложные алгоритмы для вставки"""
        return {
            "sieve_of_eratosthenes": """def sieve_of_eratosthenes(n):
    \"\"\"Решето Эратосфена - находит все простые числа до n\"\"\"
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            for multiple in range(p * p, n + 1, p):
                sieve[multiple] = False
    return [num for num in range(2, n + 1) if sieve[num]]""",

            "binary_search_impl": """def binary_search(arr, target):
    \"\"\"Двоичный поиск в отсортированном массиве\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",

            "bubble_sort_impl": """def bubble_sort(arr):
    \"\"\"Сортировка пузырьком\"\"\"
    n = len(arr)
    arr = arr.copy()  # Создаем копию, чтобы не изменять исходный
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr""",

            "quick_sort_impl": """def quick_sort(arr):
    \"\"\"Быстрая сортировка\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",

            "merge_sort_impl": """def merge_sort(arr):
    \"\"\"Сортировка слиянием\"\"\"
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result""",
        }

    def solve(self, task: str) -> str:
        """Главный метод решения"""
        try:
            self.history.append(task)
            task_lower = task.lower()

            # Определяем тип задачи
            task_type = self._identify_task(task_lower)

            # Генерируем решение
            if task_type in self.templates:
                return self.templates[task_type](task)

            # Если не распознали - умный парсер
            return self._smart_solve(task)
        except Exception as e:
            return self._error_handler(task, e)

    def _identify_task(self, task: str) -> str:
        """Определяет тип задачи по паттернам"""
        # Сначала ищем более специфичные паттерны
        priority_patterns = [
            "binary_search", "bubble_sort", "quick_sort", "merge_sort",
            "caesar_cipher", "armstrong", "palindrome", "diamond"
        ]
        
        for priority in priority_patterns:
            if priority in self.patterns:
                for pattern in self.patterns[priority]:
                    if re.search(pattern, task, re.IGNORECASE):
                        return priority
        
        # Затем все остальные
        for task_type, patterns in self.patterns.items():
            if task_type == "unknown":
                continue
            for pattern in patterns:
                if re.search(pattern, task, re.IGNORECASE):
                    return task_type
        return "unknown"

    def _extract_numbers(self, task: str) -> List[int]:
        """Извлекает все числа из строки"""
        try:
            numbers = re.findall(r'-?\d+', task)
            return [int(x) for x in numbers if x]
        except:
            return []

    def _extract_list(self, task: str) -> List:
        """Извлекает список из строки"""
        try:
            match = re.search(r'\[(.*?)\]', task)
            if match:
                items = match.group(1).split(',')
                result = []
                for item in items:
                    item = item.strip()
                    if not item:
                        continue
                    # Проверка на число
                    if item.lstrip('-').isdigit():
                        result.append(int(item))
                    elif item.lower() in ['true', 'false']:
                        result.append(item.lower() == 'true')
                    elif item.startswith('"') and item.endswith('"'):
                        result.append(item[1:-1])
                    elif item.startswith("'") and item.endswith("'"):
                        result.append(item[1:-1])
                    else:
                        result.append(item)
                return result
            return []
        except:
            return []

    def _extract_string(self, task: str) -> str:
        """Извлекает строку в кавычках"""
        try:
            match = re.search(r'["\'](.*?)["\']', task)
            return match.group(1) if match else ""
        except:
            return ""

    def _format_response(self, task: str, code: str, explanation: str, complexity: str = "средняя") -> str:
        """Форматирует ответ"""
        self.solved_count += 1
        task_type = self._identify_task(task.lower())
        return f"""
📌 ЗАДАНИЕ:
{task}

🎯 ТИП: {task_type}
⚡ СЛОЖНОСТЬ: {complexity}

💡 ОБЪЯСНЕНИЕ:
{explanation}

📝 КОД:
{code}

✅ Решено задач: {self.solved_count}
"""

    def _error_handler(self, task: str, error: Exception) -> str:
        """Обработчик ошибок"""
        return f"""
⚠️ ПРОИЗОШЛА ОШИБКА:

Задача: {task}
Ошибка: {str(error)}

🛠️ ПРЕДЛОЖЕНИЕ:
Пожалуйста, переформулируйте задачу более четко. 
Укажите:
- начальные и конечные значения
- тип операции (сумма, произведение, поиск и т.д.)
- структуры данных (списки, матрицы)
"""

    def _smart_solve(self, task: str) -> str:
        """Умный парсер для нераспознанных задач"""
        numbers = self._extract_numbers(task)
        
        if len(numbers) >= 2:
            # Возможно, это диапазон чисел
            start, end = numbers[0], numbers[1]
            
            if "сумм" in task.lower():
                return self._gen_sum_range(task)
            elif "произвед" in task.lower():
                return self._gen_product_range(task)
            elif "четн" in task.lower():
                return self._gen_even_numbers(task)
            elif "нечетн" in task.lower():
                return self._gen_odd_numbers(task)
            elif "факториал" in task.lower():
                return self._gen_factorial(task)
            elif "фибоначч" in task.lower():
                return self._gen_fibonacci(task)
        
        return self._format_response(
            task,
            "# Решение не найдено\n# Попробуйте переформулировать задачу",
            "Не удалось определить тип задачи. Уточните условие.",
            "неизвестна"
        )

    # ==================== ГЕНЕРАТОРЫ РЕШЕНИЙ ====================
    
    def _gen_print_numbers(self, task: str) -> str:
        """Вывод чисел в диапазоне"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            start, end = numbers[0], numbers[1]
            code = f"""# Вывод чисел от {start} до {end}
for i in range({start}, {end + 1}):
    print(i, end=' ')"""
            explanation = f"Цикл for проходит все числа от {start} до {end} и выводит их."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_print_range(self, task: str) -> str:
        """Вывод диапазона"""
        return self._gen_print_numbers(task)

    def _gen_sum_range(self, task: str) -> str:
        """Сумма чисел в диапазоне"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            start, end = numbers[0], numbers[1]
            code = f"""# Сумма чисел от {start} до {end}
total = 0
for i in range({start}, {end + 1}):
    total += i
print(f"Сумма чисел от {start} до {end} = {{total}}")"""
            explanation = f"Цикл for суммирует все числа от {start} до {end}."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_product_range(self, task: str) -> str:
        """Произведение чисел в диапазоне"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            start, end = numbers[0], numbers[1]
            code = f"""# Произведение чисел от {start} до {end}
product = 1
for i in range({start}, {end + 1}):
    product *= i
print(f"Произведение чисел от {start} до {end} = {{product}}")"""
            explanation = f"Цикл for перемножает все числа от {start} до {end}."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_even_numbers(self, task: str) -> str:
        """Четные числа"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            start, end = numbers[0], numbers[1]
            code = f"""# Четные числа от {start} до {end}
for i in range({start}, {end + 1}):
    if i % 2 == 0:
        print(i, end=' ')"""
            explanation = f"Цикл for проверяет каждое число на четность (остаток от деления на 2 равен 0)."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_odd_numbers(self, task: str) -> str:
        """Нечетные числа"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            start, end = numbers[0], numbers[1]
            code = f"""# Нечетные числа от {start} до {end}
for i in range({start}, {end + 1}):
    if i % 2 != 0:
        print(i, end=' ')"""
            explanation = f"Цикл for проверяет каждое число на нечетность (остаток от деления на 2 не равен 0)."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_multiples(self, task: str) -> str:
        """Кратные числа"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 3:
            divisor, start, end = numbers[0], numbers[1], numbers[2]
            code = f"""# Числа, кратные {divisor} от {start} до {end}
for i in range({start}, {end + 1}):
    if i % {divisor} == 0:
        print(i, end=' ')"""
            explanation = f"Цикл for находит числа, которые делятся на {divisor} без остатка."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_factorial(self, task: str) -> str:
        """Факториал числа"""
        numbers = self._extract_numbers(task)
        if numbers:
            n = numbers[0]
            code = f"""# Факториал числа {n}
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(f"Факториал числа {n} = {{factorial({n})}}")"""
            explanation = f"Факториал числа {n} - это произведение всех чисел от 1 до {n}."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_fibonacci(self, task: str) -> str:
        """Числа Фибоначчи"""
        numbers = self._extract_numbers(task)
        n = numbers[0] if numbers else 10
        code = f"""# Первые {n} чисел Фибоначчи
def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]

print(f"Первые {n} чисел Фибоначчи: {{fibonacci({n})}}")"""
        explanation = "Каждое следующее число Фибоначчи равно сумме двух предыдущих."
        return self._format_response(task, code, explanation, "средняя")

    def _gen_prime_numbers(self, task: str) -> str:
        """Простые числа"""
        numbers = self._extract_numbers(task)
        limit = numbers[0] if numbers else 100
        code = f"""# Простые числа до {limit}
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, {limit + 1}) if is_prime(i)]
print(f"Простые числа до {limit}: {{primes}}")"""
        explanation = f"Простое число делится только на 1 и на себя. Проверяем делители до корня из числа."
        return self._format_response(task, code, explanation, "средняя")

    def _gen_perfect_numbers(self, task: str) -> str:
        """Совершенные числа"""
        numbers = self._extract_numbers(task)
        limit = numbers[0] if numbers else 1000
        code = f"""# Совершенные числа до {limit}
def is_perfect(n):
    if n < 2:
        return False
    divisors_sum = 1
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divisors_sum += i
            if i != n // i:
                divisors_sum += n // i
    return divisors_sum == n

perfect_numbers = [i for i in range(2, {limit + 1}) if is_perfect(i)]
print(f"Совершенные числа до {limit}: {{perfect_numbers}}")"""
        explanation = "Совершенное число равно сумме всех своих делителей (кроме самого числа)."
        return self._format_response(task, code, explanation, "сложная")

    def _gen_armstrong(self, task: str) -> str:
        """Числа Армстронга"""
        numbers = self._extract_numbers(task)
        limit = numbers[0] if numbers else 1000
        code = f"""# Числа Армстронга до {limit}
def is_armstrong(n):
    digits = str(n)
    power = len(digits)
    return n == sum(int(d) ** power for d in digits)

armstrong_numbers = [i for i in range(1, {limit + 1}) if is_armstrong(i)]
print(f"Числа Армстронга до {limit}: {{armstrong_numbers}}")"""
        explanation = "Число Армстронга равно сумме своих цифр, возведенных в степень количества цифр."
        return self._format_response(task, code, explanation, "средняя")

    def _gen_palindrome(self, task: str) -> str:
        """Числа-палиндромы"""
        numbers = self._extract_numbers(task)
        limit = numbers[0] if numbers else 1000
        code = f"""# Числа-палиндромы до {limit}
def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

palindromes = [i for i in range(1, {limit + 1}) if is_palindrome(i)]
print(f"Числа-палиндромы до {limit}: {{palindromes}}")"""
        explanation = "Число-палиндром читается одинаково слева направо и справа налево."
        return self._format_response(task, code, explanation, "простая")

    def _gen_list_find(self, task: str) -> str:
        """Поиск в списке"""
        lst = self._extract_list(task)
        numbers = self._extract_numbers(task)
        target = numbers[0] if numbers else None
        
        if lst and target is not None:
            code = f"""# Поиск числа {target} в списке
arr = {lst}
found = False
for i, num in enumerate(arr):
    if num == {target}:
        print(f"Число {{target}} найдено на позиции {{i}}")
        found = True
        break
if not found:
    print(f"Число {{target}} не найдено в списке")"""
            explanation = f"Линейный поиск: проходим по списку и сравниваем каждый элемент с {target}."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_list_max(self, task: str) -> str:
        """Максимум в списке"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Поиск максимального элемента в списке
arr = {lst}
max_value = arr[0]
for num in arr:
    if num > max_value:
        max_value = num
print(f"Максимальный элемент: {{max_value}}")"""
            explanation = "Инициализируем максимум первым элементом, затем сравниваем с остальными."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_list_min(self, task: str) -> str:
        """Минимум в списке"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Поиск минимального элемента в списке
arr = {lst}
min_value = arr[0]
for num in arr:
    if num < min_value:
        min_value = num
print(f"Минимальный элемент: {{min_value}}")"""
            explanation = "Инициализируем минимум первым элементом, затем сравниваем с остальными."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_list_sum(self, task: str) -> str:
        """Сумма элементов списка"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Сумма элементов списка
arr = {lst}
total = 0
for num in arr:
    total += num
print(f"Сумма элементов: {{total}}")"""
            explanation = "Цикл for суммирует все элементы списка."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_list_average(self, task: str) -> str:
        """Среднее арифметическое списка"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Среднее арифметическое списка
arr = {lst}
total = 0
for num in arr:
    total += num
average = total / len(arr)
print(f"Среднее арифметическое: {{average:.2f}}")"""
            explanation = "Суммируем элементы и делим на их количество."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_list_sort(self, task: str) -> str:
        """Сортировка списка"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Сортировка списка (пузырьком)
arr = {lst}
n = len(arr)
for i in range(n):
    for j in range(0, n - i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
print(f"Отсортированный список: {{arr}}")"""
            explanation = "Алгоритм пузырьковой сортировки: сравниваем соседние элементы и меняем их местами."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_list_reverse(self, task: str) -> str:
        """Переворот списка"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Переворот списка
arr = {lst}
reversed_arr = []
for i in range(len(arr) - 1, -1, -1):
    reversed_arr.append(arr[i])
print(f"Исходный список: {{arr}}")
print(f"Перевернутый список: {{reversed_arr}}")"""
            explanation = "Проходим по списку с конца и добавляем элементы в новый список."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_matrix_create(self, task: str) -> str:
        """Создание матрицы"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            rows, cols = numbers[0], numbers[1]
            code = f"""# Создание матрицы {rows}x{cols}
matrix = []
for i in range({rows}):
    row = []
    for j in range({cols}):
        row.append(i * {cols} + j + 1)
    matrix.append(row)

# Вывод матрицы
for row in matrix:
    for elem in row:
        print(f"{{elem:3d}}", end=' ')
    print()"""
            explanation = f"Вложенные циклы: внешний создает строки, внутренний - столбцы матрицы {rows}x{cols}."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_multiplication_table(self, task: str) -> str:
        """Таблица умножения"""
        numbers = self._extract_numbers(task)
        n = numbers[0] if numbers else 10
        code = f"""# Таблица умножения на {n}
print(f"Таблица умножения на {{n}}:")
for i in range(1, 11):
    print(f"{{n}} x {{i}} = {{n * i}}")"""
        explanation = f"Цикл for выводит таблицу умножения для числа {n} от 1 до 10."
        return self._format_response(task, code, explanation, "простая")

    def _gen_pyramid(self, task: str) -> str:
        """Пирамида из звездочек"""
        numbers = self._extract_numbers(task)
        height = numbers[0] if numbers else 5
        code = f"""# Пирамида из звездочек высотой {height}
for i in range(1, {height} + 1):
    spaces = ' ' * ({height} - i)
    stars = '*' * (2 * i - 1)
    print(spaces + stars)"""
        explanation = f"Каждая строка содержит пробелы для центрирования и нечетное количество звездочек."
        return self._format_response(task, code, explanation, "средняя")

    def _gen_diamond(self, task: str) -> str:
        """Ромб из звездочек"""
        numbers = self._extract_numbers(task)
        height = numbers[0] if numbers else 5
        code = f"""# Ромб из звездочек высотой {height}
# Верхняя половина
for i in range(1, {height} + 1):
    spaces = ' ' * ({height} - i)
    stars = '*' * (2 * i - 1)
    print(spaces + stars)
# Нижняя половина
for i in range({height} - 1, 0, -1):
    spaces = ' ' * ({height} - i)
    stars = '*' * (2 * i - 1)
    print(spaces + stars)"""
        explanation = f"Ромб состоит из верхней пирамиды высотой {height} и перевернутой нижней."
        return self._format_response(task, code, explanation, "средняя")

    def _gen_gcd(self, task: str) -> str:
        """НОД"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
            code = f"""# НОД чисел {a} и {b}
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

result = gcd({a}, {b})
print(f"НОД чисел {{a}} и {{b}} = {{result}}")"""
            explanation = "Алгоритм Евклида: повторяем деление пока остаток не станет равен 0."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_lcm(self, task: str) -> str:
        """НОК"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
            code = f"""# НОК чисел {a} и {b}
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

result = lcm({a}, {b})
print(f"НОК чисел {{a}} и {{b}} = {{result}}")"""
            explanation = f"НОК = |a * b| / НОД(a, b). Сначала находим НОД, затем вычисляем НОК."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_binary_search(self, task: str) -> str:
        """Двоичный поиск"""
        lst = self._extract_list(task)
        numbers = self._extract_numbers(task)
        target = numbers[0] if numbers else None
        
        if lst and target is not None:
            code = f"""# Двоичный поиск числа {target} в отсортированном списке
arr = sorted({lst})  # Сначала сортируем список
target = {target}

print(f"Исходный список: {{arr}}")
print(f"Ищем число: {{target}}")

left, right = 0, len(arr) - 1
found = False

while left <= right:
    mid = (left + right) // 2
    if arr[mid] == target:
        print(f"Число найдено на позиции {{mid}}")
        found = True
        break
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1

if not found:
    print("Число не найдено")"""
            explanation = "Двоичный поиск работает на отсортированном массиве, деля диапазон поиска пополам."
            return self._format_response(task, code, explanation, "сложная")
        return self._smart_solve(task)

    def _gen_bubble_sort(self, task: str) -> str:
        """Сортировка пузырьком"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Сортировка пузырьком
arr = {lst}
n = len(arr)
print(f"Исходный список: {{arr}}")

for i in range(n):
    swapped = False
    for j in range(0, n - i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
            swapped = True
    if not swapped:
        break

print(f"Отсортированный список: {{arr}}")"""
            explanation = "Пузырьковая сортировка: сравниваем соседние элементы и меняем их местами."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_quick_sort(self, task: str) -> str:
        """Быстрая сортировка"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Быстрая сортировка
arr = {lst}
print(f"Исходный список: {{arr}}")

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

sorted_arr = quick_sort(arr)
print(f"Отсортированный список: {{sorted_arr}}")"""
            explanation = "Быстрая сортировка выбирает опорный элемент и делит массив на части."
            return self._format_response(task, code, explanation, "сложная")
        return self._smart_solve(task)

    def _gen_merge_sort(self, task: str) -> str:
        """Сортировка слиянием"""
        lst = self._extract_list(task)
        if lst:
            code = f"""# Сортировка слиянием
arr = {lst}
print(f"Исходный список: {{arr}}")

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

sorted_arr = merge_sort(arr)
print(f"Отсортированный список: {{sorted_arr}}")"""
            explanation = "Сортировка слиянием делит массив на части, сортирует их и сливает."
            return self._format_response(task, code, explanation, "сложная")
        return self._smart_solve(task)

    def _gen_string_reverse(self, task: str) -> str:
        """Переворот строки"""
        s = self._extract_string(task)
        if s:
            code = f"""# Переворот строки
s = "{s}"
reversed_s = ""
for char in s:
    reversed_s = char + reversed_s
print(f"Исходная строка: {{s}}")
print(f"Перевернутая строка: {{reversed_s}}")"""
            explanation = "Проходим по строке и добавляем каждый символ в начало новой строки."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_string_count(self, task: str) -> str:
        """Подсчет символов в строке"""
        s = self._extract_string(task)
        if s:
            code = f"""# Подсчет символов в строке
s = "{s}"
count = 0
for char in s:
    count += 1
print(f"Количество символов в строке: {{count}}")"""
            explanation = "Цикл for считает каждый символ в строке."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_string_palindrome(self, task: str) -> str:
        """Проверка строки на палиндром"""
        s = self._extract_string(task)
        if s:
            code = f"""# Проверка строки на палиндром
s = "{s}"
s_clean = ''.join(s.lower().split())
is_palindrome = True
n = len(s_clean)
for i in range(n // 2):
    if s_clean[i] != s_clean[n - 1 - i]:
        is_palindrome = False
        break

if is_palindrome:
    print(f"Строка '{{s}}' является палиндромом")
else:
    print(f"Строка '{{s}}' НЕ является палиндромом")"""
            explanation = "Сравниваем символы с начала и конца строки до середины."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_caesar_cipher(self, task: str) -> str:
        """Шифр Цезаря"""
        s = self._extract_string(task)
        numbers = self._extract_numbers(task)
        shift = numbers[0] if numbers else 3
        
        if s:
            code = f"""# Шифр Цезаря со сдвигом {shift}
def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            ascii_offset = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - ascii_offset + shift) % 26
            result += chr(ascii_offset + shifted)
        else:
            result += char
    return result

original = "{s}"
encrypted = caesar_cipher(original, {shift})
decrypted = caesar_cipher(encrypted, -{shift})

print(f"Исходная строка: {{original}}")
print(f"Зашифрованная: {{encrypted}}")
print(f"Расшифрованная: {{decrypted}}")"""
            explanation = f"Каждая буква сдвигается на {shift} позиций в алфавите."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_arithmetic_progression(self, task: str) -> str:
        """Арифметическая прогрессия"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 3:
            start, diff, count = numbers[0], numbers[1], numbers[2]
            code = f"""# Арифметическая прогрессия
a1 = {start}
d = {diff}
n = {count}

print(f"Первые {{n}} членов арифметической прогрессии:")
for i in range(n):
    term = a1 + i * d
    print(f"a{{i+1}} = {{term}}")

sum_n = n * (2 * a1 + (n - 1) * d) // 2
print(f"Сумма {{n}} членов: {{sum_n}}")"""
            explanation = f"Арифметическая прогрессия: an = a1 + (n-1)*d. Первый член: {start}, разность: {diff}."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_geometric_progression(self, task: str) -> str:
        """Геометрическая прогрессия"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 3:
            start, ratio, count = numbers[0], numbers[1], numbers[2]
            code = f"""# Геометрическая прогрессия
b1 = {start}
q = {ratio}
n = {count}

print(f"Первые {{n}} членов геометрической прогрессии:")
for i in range(n):
    term = b1 * (q ** i)
    print(f"b{{i+1}} = {{term}}")

if q != 1:
    sum_n = b1 * (q ** n - 1) // (q - 1)
    print(f"Сумма {{n}} членов: {{sum_n}}")"""
            explanation = f"Геометрическая прогрессия: bn = b1 * q^(n-1). Первый член: {start}, знаменатель: {ratio}."
            return self._format_response(task, code, explanation, "средняя")
        return self._smart_solve(task)

    def _gen_while_until(self, task: str) -> str:
        """Цикл while"""
        numbers = self._extract_numbers(task)
        if len(numbers) >= 2:
            limit = numbers[0]
            code = f"""# Цикл while до достижения {limit}
counter = 0
while counter < {limit}:
    print(f"Итерация: {{counter}}")
    counter += 1
print("Цикл завершен")"""
            explanation = f"Цикл while выполняется пока условие counter < {limit} истинно."
            return self._format_response(task, code, explanation, "простая")
        return self._smart_solve(task)

    def _gen_infinite_loop(self, task: str) -> str:
        """Бесконечный цикл"""
        code = """# Бесконечный цикл с прерыванием
counter = 0
while True:
    print(f"Итерация: {counter}")
    counter += 1
    if counter >= 10:  # Условие выхода
        print("Прерывание цикла")
        break"""
        explanation = "Бесконечный цикл while True выполняется до команды break."
        return self._format_response(task, code, explanation, "средняя")


# Пример использования
if __name__ == "__main__":
    bot = UltimateCycleBot()
    
    # Тестовые примеры
    test_tasks = [
        "выведи числа от 1 до 10",
        "найди сумму чисел от 1 до 100",
        "выведи четные числа от 1 до 20",
        "найди факториал числа 5",
        "выведи первые 10 чисел Фибоначчи",
        "найди НОД чисел 48 и 18",
        "отсортируй список [5, 2, 8, 1, 9]",
        "проверь строку 'казак' на палиндром",
        "создай матрицу 3х3",
        "найди простые числа до 50"
    ]
    
    print("=" * 60)
    print("ULTIMATE CYCLE BOT v5.1 - РЕШАТЕЛЬ ЗАДАЧ ПО ЦИКЛАМ")
    print("=" * 60)
    
    for task in test_tasks:
        print(bot.solve(task))
        print("-" * 60)
