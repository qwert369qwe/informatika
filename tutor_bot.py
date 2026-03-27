# cycle_solver.py - ИИ-решатель заданий по циклам
import re

class CycleSolver:
    def __init__(self):
        self.code_templates = []
    
    def solve(self, task):
        """Главный мозг - анализирует и решает ЛЮБОЕ задание по циклам"""
        
        # Анализируем что нужно сделать
        action = self.parse_action(task)
        numbers = self.parse_numbers(task)
        condition = self.parse_condition(task)
        target = self.parse_target(task)
        
        # Генерируем решение
        solution = self.generate_solution(action, numbers, condition, target, task)
        
        return solution
    
    def parse_action(self, task):
        """Определяет ЧТО нужно сделать"""
        t = task.lower()
        
        if any(x in t for x in ["выведи", "вывести", "покажи", "напечатай"]):
            return "print"
        if any(x in t for x in ["сумму", "сумма", "сложи", "прибавь"]):
            return "sum"
        if any(x in t for x in ["произведение", "умножь", "произведени"]):
            return "product"
        if any(x in t for x in ["найди", "найти", "поиск"]):
            return "find"
        if any(x in t for x in ["количество", "сколько", "посчитай"]):
            return "count"
        if any(x in t for x in ["среднее", "среднеарифметическое"]):
            return "average"
        if any(x in t for x in ["максимум", "максимальное", "наибольшее"]):
            return "max"
        if any(x in t for x in ["минимум", "минимальное", "наименьшее"]):
            return "min"
        if any(x in t for x in ["таблицу", "таблица"]):
            return "table"
        if any(x in t for x in ["факториал"]):
            return "factorial"
        if any(x in t for x in ["простые"]):
            return "prime"
        if any(x in t for x in ["четные", "четн"]):
            return "even"
        if any(x in t for x in ["нечетные", "нечетн"]):
            return "odd"
        if any(x in t for x in ["обратный", "наоборот", "переверни"]):
            return "reverse"
        if any(x in t for x in ["пирамид", "треугольник", "звездочк"]):
            return "pyramid"
        
        return "unknown"
    
    def parse_numbers(self, task):
        """Находит числа в задании"""
        numbers = re.findall(r'\d+', task)
        if len(numbers) == 1:
            return {"type": "single", "value": int(numbers[0])}
        elif len(numbers) >= 2:
            return {"type": "range", "start": int(numbers[0]), "end": int(numbers[1])}
        return {"type": "none"}
    
    def parse_condition(self, task):
        """Определяет условие (до, от, с шагом)"""
        t = task.lower()
        
        if "до" in t:
            match = re.search(r'до (\d+)', t)
            if match:
                return {"type": "until", "value": int(match.group(1))}
        if "от" in t and "до" in t:
            match = re.search(r'от (\d+) до (\d+)', t)
            if match:
                return {"type": "from_to", "start": int(match.group(1)), "end": int(match.group(2))}
        if "с шагом" in t or "шагом" in t:
            match = re.search(r'шагом (\d+)', t)
            if match:
                return {"type": "step", "value": int(match.group(1))}
        
        return {"type": "none"}
    
    def parse_target(self, task):
        """Определяет цель (список, число, что ищем)"""
        t = task.lower()
        
        # Ищем список в задании [1, 2, 3]
        list_match = re.search(r'\[(.*?)\]', task)
        if list_match:
            items = [int(x.strip()) for x in list_match.group(1).split(',') if x.strip().isdigit()]
            return {"type": "list", "value": items}
        
        # Ищем конкретное число
        num_match = re.search(r'число (\d+)', t)
        if num_match:
            return {"type": "number", "value": int(num_match.group(1))}
        
        return {"type": "none"}
    
    def generate_solution(self, action, numbers, condition, target, task):
        """Генерирует код и объяснение"""
        
        # БАЗОВЫЕ СЛУЧАИ
        
        # 1. ВЫВЕСТИ ЧИСЛА
        if action == "print":
            if numbers["type"] == "range":
                start, end = numbers["start"], numbers["end"]
                if condition.get("step"):
                    step = condition["value"]
                    code = f"for i in range({start}, {end + 1}, {step}):\n    print(i)"
                    explanation = f"Цикл for проходит от {start} до {end} с шагом {step} и выводит каждое число"
                else:
                    code = f"for i in range({start}, {end + 1}):\n    print(i)"
                    explanation = f"Цикл for проходит от {start} до {end} и выводит каждое число"
            elif numbers["type"] == "single":
                end = numbers["value"]
                code = f"for i in range(1, {end + 1}):\n    print(i)"
                explanation = f"Цикл for выводит числа от 1 до {end}"
            else:
                code = "for i in range(1, 11):\n    print(i)"
                explanation = "Цикл for выводит числа от 1 до 10"
        
        # 2. СУММА
        elif action == "sum":
            if numbers["type"] == "range":
                start, end = numbers["start"], numbers["end"]
                code = f"""summa = 0
for i in range({start}, {end + 1}):
    summa += i
print(f"Сумма чисел от {start} до {end} = {{summa}}")"""
                explanation = f"Создаём переменную summa, в цикле прибавляем каждое число от {start} до {end}"
            else:
                code = """summa = 0
for i in range(1, 101):
    summa += i
print(f"Сумма чисел от 1 до 100 = {summa}")"""
                explanation = "Суммируем все числа от 1 до 100"
        
        # 3. ЧЕТНЫЕ
        elif action == "even":
            if numbers["type"] == "range":
                end = numbers.get("end", numbers.get("value", 20))
                code = f"""for i in range(0, {end + 1}, 2):
    print(i)"""
                explanation = f"range с шагом 2 выводит только чётные числа от 0 до {end}"
            else:
                code = """for i in range(0, 21, 2):
    print(i)"""
                explanation = "range с шагом 2 выводит чётные числа"
        
        # 4. НЕЧЕТНЫЕ
        elif action == "odd":
            if numbers["type"] == "range":
                end = numbers.get("end", numbers.get("value", 20))
                code = f"""for i in range(1, {end + 1}, 2):
    print(i)"""
                explanation = f"range с шагом 2, начиная с 1, выводит нечётные числа"
            else:
                code = """for i in range(1, 21, 2):
    print(i)"""
                explanation = "range с шагом 2, начиная с 1, выводит нечётные числа"
        
        # 5. ПОИСК В СПИСКЕ
        elif action == "find" and target["type"] == "list":
            items = target["value"]
            search = numbers.get("value", 7)
            code = f"""numbers = {items}
search = {search}
found = False
for num in numbers:
    if num == search:
        print(f"Нашли число {{search}}!")
        found = True
        break
if not found:
    print(f"Число {{search}} не найдено")"""
            explanation = f"Перебираем список, ищем число {search}, если находим — выводим и выходим"
        
        # 6. ФАКТОРИАЛ
        elif action == "factorial":
            n = numbers.get("value", 5)
            code = f"""n = {n}
factorial = 1
for i in range(1, n + 1):
    factorial *= i
print(f"Факториал числа {n} = {{factorial}}")"""
            explanation = f"Факториал — произведение чисел от 1 до {n}"
        
        # 7. ТАБЛИЦА УМНОЖЕНИЯ
        elif action == "table":
            num = numbers.get("value", 5)
            code = f"""num = {num}
for i in range(1, 11):
    print(f"{{num}} × {{i}} = {{num * i}}")"""
            explanation = f"Таблица умножения на {num}"
        
        # 8. МАКСИМУМ/МИНИМУМ
        elif action == "max" and target["type"] == "list":
            items = target["value"]
            code = f"""numbers = {items}
maximum = numbers[0]
for num in numbers:
    if num > maximum:
        maximum = num
print(f"Максимальное число: {{maximum}}")"""
            explanation = "Идём по списку, запоминаем самое большое число"
        
        elif action == "min" and target["type"] == "list":
            items = target["value"]
            code = f"""numbers = {items}
minimum = numbers[0]
for num in numbers:
    if num < minimum:
        minimum = num
print(f"Минимальное число: {{minimum}}")"""
            explanation = "Идём по списку, запоминаем самое маленькое число"
        
        # 9. КОЛИЧЕСТВО
        elif action == "count":
            if target["type"] == "number":
                search = target["value"]
                if numbers["type"] == "range":
                    start, end = numbers["start"], numbers["end"]
                    code = f"""count = 0
for i in range({start}, {end + 1}):
    if i == {search}:
        count += 1
print(f"Число {search} встречается {{count}} раз")"""
                    explanation = f"Считаем сколько раз число {search} встречается в диапазоне"
            else:
                code = """count = 0
for i in range(1, 101):
    if i % 2 == 0:
        count += 1
print(f"Количество чётных чисел: {count}")"""
                explanation = "Считаем количество чётных чисел"
        
        # 10. ПИРАМИДКА
        elif action == "pyramid":
            height = numbers.get("value", 5)
            code = f"""height = {height}
for i in range(1, height + 1):
    print('*' * i)"""
            explanation = f"Каждая строка содержит i звёздочек, i от 1 до {height}"
        
        # 11. ОБРАТНЫЙ ПОРЯДОК
        elif action == "reverse":
            if numbers["type"] == "range":
                start, end = numbers["start"], numbers["end"]
                code = f"""for i in range({end}, {start - 1}, -1):
    print(i)"""
                explanation = f"range с отрицательным шагом выводит числа от {end} до {start}"
            else:
                code = """for i in range(10, 0, -1):
    print(i)"""
                explanation = "range с шагом -1 выводит числа в обратном порядке"
        
        # 12. ПРОСТЫЕ ЧИСЛА
        elif action == "prime":
            limit = numbers.get("value", 20)
            code = f"""limit = {limit}
for num in range(2, limit + 1):
    is_prime = True
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        print(num)"""
            explanation = f"Проверяем каждое число от 2 до {limit} на простоту"
        
        # 13. WHILE ДО УСЛОВИЯ
        elif action == "unknown" and "пока" in task.lower():
            code = """i = 1
while i <= 10:
    print(i)
    i += 1"""
            explanation = "while выполняется пока условие i <= 10 истинно"
        
        # 14. UNKNOWN — ПЫТАЕМСЯ УГАДАТЬ
        else:
            # Анализируем по ключевым словам
            t = task.lower()
            if "цикл" in t:
                code = """for i in range(10):
    print(i)"""
                explanation = "Стандартный цикл for для 10 итераций"
            elif "while" in t:
                code = """i = 0
while i < 10:
    print(i)
    i += 1"""
                explanation = "Стандартный цикл while"
            else:
                code = """# Я понял задание так:
for i in range(1, 11):
    print(i)
# Если не то, напиши задание подробнее"""
                explanation = "Я попробовал угадать задание. Уточни, если не то"
        
        # Формируем ответ
        return f"""
╔══════════════════════════════════════════════════════════╗
║  🧠 РЕШЕНИЕ ЗАДАНИЯ                                      ║
╚══════════════════════════════════════════════════════════╝

📌 ЗАДАНИЕ:
{task}

💡 ОБЪЯСНЕНИЕ:
{explanation}

📝 КОД:
```python
{code}
