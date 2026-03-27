# cycle_genius.py - ИИ-генератор кода по циклам
# Версия 4.0 - полный парсинг, анализ и генерация без шаблонов

import re
import random

class CycleGenius:
    def __init__(self):
        self.understood = False
        
    def think(self, task):
        """Главный мозг - анализирует и генерирует решение"""
        
        # ШАГ 1: РАЗБИРАЕМ ЗАДАНИЕ НА ЧАСТИ
        task_lower = task.lower()
        
        # Определяем тип действия
        action = self._get_action(task_lower)
        
        # Находим все числа
        numbers = self._get_numbers(task)
        
        # Находим диапазоны
        ranges = self._get_ranges(task_lower)
        
        # Находим списки
        lists = self._get_lists(task)
        
        # Находим условия
        conditions = self._get_conditions(task_lower)
        
        # Определяем тип цикла
        loop_type = self._choose_loop(task_lower, action, ranges)
        
        # Определяем направление
        direction = self._get_direction(task_lower)
        
        # Определяем шаг
        step = self._get_step(task_lower)
        
        # Определяем оператор (break/continue)
        special_op = self._get_special_operator(task_lower)
        
        # ШАГ 2: ГЕНЕРИРУЕМ КОД
        code = self._generate_code(
            action, numbers, ranges, lists, conditions, 
            loop_type, direction, step, special_op, task_lower
        )
        
        # ШАГ 3: ГЕНЕРИРУЕМ ОБЪЯСНЕНИЕ
        explanation = self._explain(
            action, numbers, ranges, lists, conditions,
            loop_type, direction, step, special_op
        )
        
        # ШАГ 4: ВОЗВРАЩАЕМ РЕЗУЛЬТАТ
        return self._format_output(task, code, explanation)
    
    def _get_action(self, task):
        """Определяет действие"""
        actions = {
            'print': ['выведи', 'вывести', 'покажи', 'напечатай', 'вывод'],
            'sum': ['сумму', 'сумма', 'сложи', 'прибавь', 'сложить'],
            'product': ['произведение', 'умножь', 'умножить', 'произведени'],
            'count': ['количество', 'сколько', 'посчитай', 'подсчитай'],
            'average': ['среднее', 'среднеарифметическое', 'средняя'],
            'max': ['максимум', 'максимальное', 'наибольшее', 'самое большое'],
            'min': ['минимум', 'минимальное', 'наименьшее', 'самое маленькое'],
            'find': ['найди', 'найти', 'поиск', 'отыщи'],
            'table': ['таблиц', 'таблица'],
            'factorial': ['факториал'],
            'prime': ['простое', 'простые'],
            'even': ['четн'],
            'odd': ['нечетн'],
            'reverse': ['обратн', 'наоборот', 'переверн'],
            'pyramid': ['пирамид', 'треугольник', 'звездочк', 'ёлочк'],
            'fibonacci': ['фибоначч', 'фибоначи'],
            'palindrome': ['палиндром'],
            'gcd': ['нод', 'наибольший общий делитель', 'gcd'],
            'lcm': ['нок', 'наименьшее общее кратное', 'lcm'],
            'sort': ['сортир', 'отсортир', 'упорядоч'],
            'filter': ['фильтр', 'отфильтр', 'оставь'],
            'map': ['преобразуй', 'измени'],
            'reduce': ['сверни', 'сократи'],
        }
        
        for action, keywords in actions.items():
            if any(k in task for k in keywords):
                return action
        return 'unknown'
    
    def _get_numbers(self, task):
        """Извлекает все числа из задания"""
        numbers = re.findall(r'\d+', task)
        return [int(n) for n in numbers]
    
    def _get_ranges(self, task):
        """Находит диапазоны типа 'от X до Y', 'с X по Y'"""
        ranges = []
        
        # от X до Y
        match = re.search(r'от (\d+) до (\d+)', task)
        if match:
            ranges.append(('range', int(match.group(1)), int(match.group(2))))
        
        # с X по Y
        match = re.search(r'с (\d+) по (\d+)', task)
        if match:
            ranges.append(('range', int(match.group(1)), int(match.group(2))))
        
        # от X
        match = re.search(r'от (\d+)', task)
        if match and not ranges:
            ranges.append(('from', int(match.group(1)), None))
        
        # до Y
        match = re.search(r'до (\d+)', task)
        if match and not ranges:
            ranges.append(('to', None, int(match.group(1))))
        
        return ranges
    
    def _get_lists(self, task):
        """Извлекает списки из задания"""
        lists = []
        # Ищем [1, 2, 3]
        match = re.search(r'\[(.*?)\]', task)
        if match:
            items = [x.strip() for x in match.group(1).split(',')]
            # Пробуем преобразовать в числа
            try:
                items = [int(x) for x in items if x.strip().isdigit()]
            except:
                items = [x.strip(' "\'') for x in items]
            lists.append(items)
        return lists
    
    def _get_conditions(self, task):
        """Находит условия"""
        conditions = []
        
        if 'кратный' in task or 'делится' in task:
            nums = self._get_numbers(task)
            if nums:
                conditions.append(('multiple', nums[0]))
        
        if 'больше' in task:
            nums = self._get_numbers(task)
            if nums:
                conditions.append(('greater', nums[0]))
        
        if 'меньше' in task:
            nums = self._get_numbers(task)
            if nums:
                conditions.append(('less', nums[0]))
        
        if 'равен' in task or 'равно' in task:
            nums = self._get_numbers(task)
            if nums:
                conditions.append(('equal', nums[0]))
        
        return conditions
    
    def _choose_loop(self, task, action, ranges):
        """Выбирает тип цикла"""
        # Если есть while в задании
        if 'while' in task:
            return 'while'
        
        # Если есть условие типа "пока не"
        if 'пока не' in task:
            return 'while'
        
        # Если есть бесконечность
        if 'бесконеч' in task:
            return 'while'
        
        # Если есть чётко заданный диапазон
        if ranges:
            return 'for'
        
        # Если действие связано с перебором
        if action in ['print', 'sum', 'product', 'count', 'average', 'max', 'min', 'filter', 'map']:
            return 'for'
        
        # По умолчанию for
        return 'for'
    
    def _get_direction(self, task):
        """Определяет направление"""
        if 'обратн' in task or 'наоборот' in task or 'убыва' in task:
            return 'desc'
        if 'возраст' in task:
            return 'asc'
        return 'asc'
    
    def _get_step(self, task):
        """Находит шаг"""
        match = re.search(r'шагом (\d+)', task)
        if match:
            return int(match.group(1))
        
        if 'через один' in task or 'через 1' in task:
            return 2
        
        return 1
    
    def _get_special_operator(self, task):
        """Определяет нужен ли break или continue"""
        if 'прерв' in task or 'останов' in task or 'выйд' in task:
            return 'break'
        if 'пропуст' in task or 'скип' in task:
            return 'continue'
        return None
    
    def _generate_code(self, action, numbers, ranges, lists, conditions, 
                       loop_type, direction, step, special_op, task):
        """Генерирует код на основе анализа"""
        
        code_lines = []
        variables = {}
        
        # Определяем переменные
        if lists:
            var_name = 'data'
            code_lines.append(f"{var_name} = {lists[0]}")
            iterable = var_name
        elif ranges:
            if ranges[0][0] == 'range':
                start, end = ranges[0][1], ranges[0][2]
                if direction == 'desc':
                    iterable = f"range({end}, {start-1}, -{step})"
                else:
                    iterable = f"range({start}, {end+1}, {step})"
            elif ranges[0][0] == 'from':
                start = ranges[0][1]
                if direction == 'desc':
                    iterable = f"range({start}, 0, -{step})"
                else:
                    iterable = f"range({start}, {start+10}, {step})"
            elif ranges[0][0] == 'to':
                end = ranges[0][2]
                iterable = f"range(1, {end+1}, {step})"
        elif numbers:
            if len(numbers) == 1:
                iterable = f"range(1, {numbers[0]+1}, {step})"
            elif len(numbers) >= 2:
                if direction == 'desc':
                    iterable = f"range({numbers[1]}, {numbers[0]-1}, -{step})"
                else:
                    iterable = f"range({numbers[0]}, {numbers[1]+1}, {step})"
        else:
            iterable = "range(1, 11, step)" if step > 1 else "range(1, 11)"
        
        # Создаём цикл
        if loop_type == 'for':
            code_lines.append(f"for i in {iterable}:")
            indent = "    "
        else:
            # while цикл
            if 'бесконеч' in task:
                code_lines.append("while True:")
            else:
                counter = 'counter'
                code_lines.append(f"{counter} = 0")
                if ranges:
                    end = ranges[0][2] if ranges[0][2] else 10
                    code_lines.append(f"while {counter} < {end}:")
                else:
                    code_lines.append(f"while {counter} < 10:")
            indent = "    "
        
        # Добавляем break/continue если нужно
        if special_op == 'break' and conditions:
            cond = conditions[0]
            if cond[0] == 'equal':
                code_lines.append(f"{indent}if i == {cond[1]}:")
                code_lines.append(f"{indent}    break")
        elif special_op == 'continue' and conditions:
            cond = conditions[0]
            if cond[0] == 'multiple':
                code_lines.append(f"{indent}if i % {cond[1]} == 0:")
                code_lines.append(f"{indent}    continue")
        
        # Генерируем тело цикла в зависимости от действия
        if action == 'print':
            code_lines.append(f"{indent}print(i)")
        
        elif action == 'sum':
            code_lines.insert(0, "total = 0")
            code_lines.append(f"{indent}total += i")
            code_lines.append("print(f'Сумма = {total}')")
        
        elif action == 'product':
            code_lines.insert(0, "product = 1")
            code_lines.append(f"{indent}product *= i")
            code_lines.append("print(f'Произведение = {product}')")
        
        elif action == 'count':
            code_lines.insert(0, "count = 0")
            if conditions:
                cond = conditions[0]
                if cond[0] == 'multiple':
                    code_lines.append(f"{indent}if i % {cond[1]} == 0:")
                    code_lines.append(f"{indent}    count += 1")
                else:
                    code_lines.append(f"{indent}count += 1")
            else:
                code_lines.append(f"{indent}count += 1")
            code_lines.append("print(f'Количество = {count}')")
        
        elif action == 'average':
            code_lines.insert(0, "total = 0")
            code_lines.insert(1, "count = 0")
            code_lines.append(f"{indent}total += i")
            code_lines.append(f"{indent}count += 1")
            code_lines.append("print(f'Среднее = {total / count}')")
        
        elif action == 'max':
            code_lines.insert(0, "maximum = float('-inf')")
            code_lines.append(f"{indent}if i > maximum:")
            code_lines.append(f"{indent}    maximum = i")
            code_lines.append("print(f'Максимум = {maximum}')")
        
        elif action == 'min':
            code_lines.insert(0, "minimum = float('inf')")
            code_lines.append(f"{indent}if i < minimum:")
            code_lines.append(f"{indent}    minimum = i")
            code_lines.append("print(f'Минимум = {minimum}')")
        
        elif action == 'find':
            if lists:
                code_lines.append(f"{indent}if i == target:")
                code_lines.append(f"{indent}    print('Нашли!')")
                code_lines.append(f"{indent}    break")
            else:
                code_lines.append(f"{indent}if i == target:")
                code_lines.append(f"{indent}    print('Нашли!')")
                code_lines.append(f"{indent}    break")
        
        elif action == 'even':
            code_lines.append(f"{indent}if i % 2 == 0:")
            code_lines.append(f"{indent}    print(i)")
        
        elif action == 'odd':
            code_lines.append(f"{indent}if i % 2 != 0:")
            code_lines.append(f"{indent}    print(i)")
        
        elif action == 'reverse':
            code_lines.append(f"{indent}print(i)")
        
        elif action == 'pyramid':
            code_lines = [f"for i in range(1, {numbers[0] if numbers else 5} + 1):"]
            code_lines.append(f"{indent}print('*' * i)")
        
        elif action == 'fibonacci':
            n = numbers[0] if numbers else 10
            code_lines = [
                f"a, b = 0, 1",
                f"for i in range({n}):",
                f"{indent}print(a)",
                f"{indent}a, b = b, a + b"
            ]
        
        elif action == 'factorial':
            n = numbers[0] if numbers else 5
            code_lines = [
                f"factorial = 1",
                f"for i in range(1, {n} + 1):",
                f"{indent}factorial *= i",
                f"print(f'Факториал {n} = {{factorial}}')"
            ]
        
        elif action == 'prime':
            limit = numbers[0] if numbers else 20
            code_lines = [
                f"for num in range(2, {limit} + 1):",
                f"{indent}is_prime = True",
                f"{indent}for i in range(2, int(num ** 0.5) + 1):",
                f"{indent}    if num % i == 0:",
                f"{indent}        is_prime = False",
                f"{indent}        break",
                f"{indent}if is_prime:",
                f"{indent}    print(num)"
            ]
        
        elif action == 'table':
            num = numbers[0] if numbers else 5
            code_lines = [
                f"for i in range(1, 11):",
                f"{indent}print(f'{num} × {{i}} = {num * i}')"
            ]
        
        elif action == 'filter':
            if conditions:
                cond = conditions[0]
                if cond[0] == 'multiple':
                    code_lines.append(f"{indent}if i % {cond[1]} == 0:")
                    code_lines.append(f"{indent}    print(i)")
            else:
                code_lines.append(f"{indent}print(i)")
        
        else:
            code_lines.append(f"{indent}print(i)")
        
        # Добавляем инкремент для while
        if loop_type == 'while' and 'бесконеч' not in task:
            code_lines.insert(1, f"{indent}{counter} += 1")
        
        return '\n'.join(code_lines)
    
    def _explain(self, action, numbers, ranges, lists, conditions, 
                 loop_type, direction, step, special_op):
        """Генерирует объяснение"""
        parts = []
        
        parts.append(f"🔹 Тип цикла: {loop_type.upper()}")
        
        if ranges:
            parts.append(f"🔹 Диапазон: от {ranges[0][1]} до {ranges[0][2]}")
        elif numbers:
            parts.append(f"🔹 Числа в задании: {numbers}")
        
        if lists:
            parts.append(f"🔹 Список: {lists[0]}")
        
        if conditions:
            parts.append(f"🔹 Условие: {conditions[0]}")
        
        if direction == 'desc':
            parts.append(f"🔹 Направление: обратный порядок")
        
        if step > 1:
            parts.append(f"🔹 Шаг: {step}")
        
        if special_op:
            parts.append(f"🔹 Специальный оператор: {special_op}")
        
        action_names = {
            'print': 'вывод чисел',
            'sum': 'суммирование чисел',
            'product': 'произведение чисел',
            'count': 'подсчёт количества',
            'average': 'вычисление среднего',
            'max': 'поиск максимума',
            'min': 'поиск минимума',
            'find': 'поиск элемента',
            'even': 'вывод чётных чисел',
            'odd': 'вывод нечётных чисел',
            'reverse': 'вывод в обратном порядке',
            'pyramid': 'построение пирамиды',
            'fibonacci': 'генерация чисел Фибоначчи',
            'factorial': 'вычисление факториала',
            'prime': 'поиск простых чисел',
            'table': 'таблица умножения',
            'filter': 'фильтрация элементов',
        }
        
        parts.append(f"🔹 Действие: {action_names.get(action, 'обработка данных')}")
        
        return '\n'.join(parts)
    
    def _format_output(self, task, code, explanation):
        """Форматирует ответ"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              🧠 ГЕНИЙ ЦИКЛОВ - РЕШЕНИЕ ЗАДАНИЯ               ║
╚══════════════════════════════════════════════════════════════╝

📌 ВАШЕ ЗАДАНИЕ:
{task}

🔍 АНАЛИЗ ЗАДАНИЯ:
{explanation}

💻 СГЕНЕРИРОВАННЫЙ КОД:
```python
{code}
