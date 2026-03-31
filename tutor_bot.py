# full_cycles_guide.py - ПОЛНЫЙ СПРАВОЧНИК ПО ЦИКЛАМ
# Запусти и получи всю теорию + примеры

import sys

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_code(code):
    print("```python")
    print(code)
    print("```")

def main():
    print("\n" + "█" * 70)
    print("     ПОЛНЫЙ СПРАВОЧНИК ПО ЦИКЛАМ В PYTHON")
    print("     ВСЯ ТЕОРИЯ + ВСЕ ПРИМЕРЫ РЕШЕНИЙ")
    print("█" * 70)
    
    # ==================== 1. ЦИКЛ FOR ====================
    print_header("1. ЦИКЛ FOR - основы")
    print("""
🔹 СИНТАКСИС:
   for переменная in последовательность:
       # действие

🔹 КЛЮЧЕВЫЕ ПОНЯТИЯ:
   • Перебирает элементы последовательности (список, строка, range)
   • Количество итераций = длине последовательности
   • Переменная создаётся автоматически

🔹 ПРИМЕРЫ:
   1) Простой перебор:
   for i in range(5):
       print(i)  # 0,1,2,3,4

   2) Перебор списка:
   fruits = ['яблоко', 'банан', 'апельсин']
   for fruit in fruits:
       print(fruit)

   3) Перебор строки:
   for letter in 'Python':
       print(letter)
""")
    
    # ==================== 2. ФУНКЦИЯ RANGE ====================
    print_header("2. ФУНКЦИЯ RANGE()")
    print("""
🔹 СИНТАКСИС:
   range(stop)                 # от 0 до stop-1
   range(start, stop)          # от start до stop-1
   range(start, stop, step)    # от start до stop-1 с шагом step

🔹 ПРИМЕРЫ:
   range(5)        → 0,1,2,3,4
   range(2,8)      → 2,3,4,5,6,7
   range(0,10,2)   → 0,2,4,6,8
   range(10,0,-1)  → 10,9,8,7,6,5,4,3,2,1

🔹 В ЦИКЛЕ:
   for i in range(1, 11):
       print(i)  # числа от 1 до 10
""")
    
    # ==================== 3. ЦИКЛ WHILE ====================
    print_header("3. ЦИКЛ WHILE")
    print("""
🔹 СИНТАКСИС:
   while условие:
       # действие
       # изменение условия (обязательно!)

🔹 ВАЖНО:
   • Выполняется, пока условие истинно
   • Если условие всегда True → бесконечный цикл
   • Нужно обязательно менять условие внутри цикла

🔹 ПРИМЕРЫ:
   1) Счётчик:
   i = 0
   while i < 5:
       print(i)
       i += 1  # обязательно!

   2) Сумма чисел:
   total = 0
   i = 1
   while i <= 100:
       total += i
       i += 1
   print(total)  # 5050

   3) Бесконечный цикл с break:
   while True:
       answer = input("Введите 'выход': ")
       if answer == 'выход':
           break
       print(f"Вы ввели: {answer}")
""")
    
    # ==================== 4. BREAK И CONTINUE ====================
    print_header("4. ОПЕРАТОРЫ BREAK И CONTINUE")
    print("""
🔹 BREAK - прерывает цикл полностью:
   for i in range(10):
       if i == 5:
           break
       print(i)  # выведет 0,1,2,3,4

🔹 CONTINUE - пропускает текущую итерацию:
   for i in range(5):
       if i == 2:
           continue
       print(i)  # выведет 0,1,3,4

🔹 СРАВНЕНИЕ:
   • break - выходит из цикла
   • continue - переходит к следующему шагу
""")
    
    # ==================== 5. ELSE В ЦИКЛАХ ====================
    print_header("5. ELSE В ЦИКЛАХ")
    print("""
🔹 СИНТАКСИС:
   for переменная in последовательность:
       # тело цикла
       if условие:
           break
   else:
       # выполнится, если break не сработал

🔹 ПРИМЕР:
   numbers = [1, 2, 3, 4, 5]
   search = 6
   for num in numbers:
       if num == search:
           print("Нашли!")
           break
   else:
       print("Не нашли")  # выполнится, т.к. break не сработал
""")
    
    # ==================== 6. ВЛОЖЕННЫЕ ЦИКЛЫ ====================
    print_header("6. ВЛОЖЕННЫЕ ЦИКЛЫ")
    print("""
🔹 ОПИСАНИЕ:
   Цикл внутри цикла. Внутренний цикл выполняется полностью 
   для каждой итерации внешнего.

🔹 ПРИМЕР - таблица умножения:
   for i in range(1, 4):
       for j in range(1, 4):
           print(f"{i} × {j} = {i*j}")
       print()  # пустая строка после каждой строки

🔹 ПРИМЕР - матрица:
   matrix = []
   for i in range(3):
       row = []
       for j in range(4):
           row.append(i * 4 + j + 1)
       matrix.append(row)
   
   for row in matrix:
       print(row)
""")
    
    # ==================== 7. ПРИМЕРЫ ЗАДАЧ ====================
    print_header("7. ПРИМЕРЫ РЕШЕНИЯ ЗАДАЧ")
    
    print("""
🔹 ЗАДАЧА 1: Вывести числа от 1 до 20
   РЕШЕНИЕ:
   for i in range(1, 21):
       print(i)

🔹 ЗАДАЧА 2: Найти сумму чисел от 1 до 100
   РЕШЕНИЕ:
   total = 0
   for i in range(1, 101):
       total += i
   print(total)

🔹 ЗАДАЧА 3: Вывести чётные числа от 0 до 20
   РЕШЕНИЕ:
   for i in range(0, 21, 2):
       print(i)

🔹 ЗАДАЧА 4: Найти факториал числа 5 (5! = 1×2×3×4×5)
   РЕШЕНИЕ:
   n = 5
   factorial = 1
   for i in range(1, n + 1):
       factorial *= i
   print(factorial)

🔹 ЗАДАЧА 5: Вывести таблицу умножения на 5
   РЕШЕНИЕ:
   for i in range(1, 11):
       print(f"5 × {i} = {5*i}")

🔹 ЗАДАЧА 6: Найти число 7 в списке [3,8,1,7,4,9]
   РЕШЕНИЕ:
   numbers = [3, 8, 1, 7, 4, 9]
   for num in numbers:
       if num == 7:
           print("Нашёл!")
           break

🔹 ЗАДАЧА 7: Вывести числа Фибоначчи до 10
   РЕШЕНИЕ:
   a, b = 0, 1
   for i in range(10):
       print(a)
       a, b = b, a + b

🔹 ЗАДАЧА 8: Найти простые числа до 50
   РЕШЕНИЕ:
   for num in range(2, 51):
       is_prime = True
       for i in range(2, int(num**0.5) + 1):
           if num % i == 0:
               is_prime = False
               break
       if is_prime:
           print(num)

🔹 ЗАДАЧА 9: Вывести пирамидку из звёздочек высотой 5
   РЕШЕНИЕ:
   for i in range(1, 6):
       print('*' * i)

🔹 ЗАДАЧА 10: Найти максимальное число в списке [64,34,25,12,22,11,90]
   РЕШЕНИЕ:
   numbers = [64, 34, 25, 12, 22, 11, 90]
   maximum = numbers[0]
   for num in numbers:
       if num > maximum:
           maximum = num
   print(maximum)

🔹 ЗАДАЧА 11: Перевернуть строку "Python"
   РЕШЕНИЕ:
   text = "Python"
   reversed_text = ""
   for char in text:
       reversed_text = char + reversed_text
   print(reversed_text)

🔹 ЗАДАЧА 12: Проверить, является ли число палиндромом (12321)
   РЕШЕНИЕ:
   num = 12321
   if str(num) == str(num)[::-1]:
       print("Палиндром")

🔹 ЗАДАЧА 13: Найти НОД чисел 48 и 180
   РЕШЕНИЕ:
   a, b = 48, 180
   while b:
       a, b = b, a % b
   print(a)

🔹 ЗАДАЧА 14: Найти сумму всех элементов списка [1,2,3,4,5]
   РЕШЕНИЕ:
   numbers = [1, 2, 3, 4, 5]
   total = 0
   for num in numbers:
       total += num
   print(total)

🔹 ЗАДАЧА 15: Вывести числа от 10 до 1 в обратном порядке
   РЕШЕНИЕ:
   for i in range(10, 0, -1):
       print(i)
""")
    
    # ==================== 8. ШПАРГАЛКА ====================
    print_header("8. КРАТКАЯ ШПАРГАЛКА")
    print("""
┌─────────────────────────────────────────────────────────────────┐
│  for i in range(10):           # 0..9                          │
│  for i in range(1, 11):        # 1..10                         │
│  for i in range(0, 10, 2):     # 0,2,4,6,8                     │
│  for i in range(10, 0, -1):    # 10,9,8,...,1                  │
│                                                                │
│  for item in список:           # перебор списка                │
│  for key, val in dict.items(): # перебор словаря              │
│  for i, item in enumerate(list): # с индексом                  │
│                                                                │
│  while условие:                 # цикл с условием               │
│      действие                                                  │
│      изменение_условия                                         │
│                                                                │
│  while True:                    # бесконечный цикл              │
│      if условие:                                               │
│          break                  # выход                         │
│                                                                │
│  break                          # прервать цикл                 │
│  continue                       # пропустить итерацию           │
│                                                                │
│  for i in range(10):            # else выполнится если         │
│      if i == 5:                 # не было break                 │
│          break                                                 │
│  else:                                                         │
│      print("не было break")                                    │
└─────────────────────────────────────────────────────────────────┘
""")
    
    print("\n" + "█" * 70)
    print("     ГОТОВО! ВСЯ ТЕОРИЯ И ПРИМЕРЫ ПЕРЕД ТОБОЙ")
    print("     СОХРАНИ ЭТОТ ФАЙЛ КАК ШПАРГАЛКУ")
    print("█" * 70)

if __name__ == "__main__":
    main()
