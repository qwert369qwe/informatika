# ==================== ОДНОСТРОЧНЫЙ РЕЖИМ ====================
if __name__ == '__main__':
    import sys
    
    # Если передан аргумент командной строки
    if len(sys.argv) > 1:
        task = sys.argv[1]
        print(f"🤖 Задача: {task}")
        
        # Создаём интерфейс (модель подгрузится сама)
        interface = ImprovedNeuralInterface()
        
        # Решаем задачу
        solution = interface.solve_task(task)
        
    else:
        print("❌ Ошибка: не указана задача")
        print("Использование: python neiroset.py 'твоя задача'")
