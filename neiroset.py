import requests

# Первая строка - промт (как нужно решать)
prompt = input()
# Вторая строка - задача
task = input()

# Формируем полный запрос
full_prompt = f"{prompt} {task}"

url = "https://text.pollinations.ai/"
response = requests.get(f"{url}{full_prompt}")

code = response.text
print("\n=== Код ===\n")
print(code)
print("\n=== Результат ===\n")
exec(code)
