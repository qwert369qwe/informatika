# tutor_bot.py - Репетитор по циклам
print("🤖 Бот-репетитор по циклам запущен!")
print("Спрашивай про: for, while, range, break, continue")
print('Напиши "выход" чтобы выйти')
print("-" * 40)

while True:
    question = input("\n👨‍🎓 Твой вопрос: ")
    
    if question.lower() == "выход":
        print("👋 Удачи на самостоятельной!")
        break
    
    q = question.lower()
    
    if "for" in q:
        print("""
📚 Цикл for:
for i in range(5):
    print(i)  # выведет 0,1,2,3,4

for fruit in ['яблоко', 'банан']:
    print(fruit)
""")
    elif "while" in q:
        print("""
📚 Цикл while:
i = 0
while i < 5:
    print(i)
    i += 1  # важно! иначе бесконечный цикл
""")
    elif "range" in q:
        print("""
📚 Функция range():
range(5)     -> 0,1,2,3,4
range(1,6)   -> 1,2,3,4,5
range(0,10,2) -> 0,2,4,6,8
""")
    elif "break" in q:
        print("""
📚 break - прерывает цикл:
for i in range(10):
    if i == 5:
        break
    print(i)  # 0,1,2,3,4
""")
    elif "continue" in q:
        print("""
📚 continue - пропускает итерацию:
for i in range(5):
    if i == 2:
        continue
    print(i)  # 0,1,3,4
""")
    else:
        print("""
Я могу рассказать о:
• for - цикл для перебора
• while - цикл с условием
• range - диапазон чисел
• break - прервать цикл
• continue - пропустить шаг

Просто напиши одно из этих слов!
""")