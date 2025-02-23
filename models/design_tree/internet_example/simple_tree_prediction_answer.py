class DecisionNode:
    def __init__(self, question, yes_answer=None, no_answer=None):
        self.question = question  # Вопрос или условие
        self.yes_answer = yes_answer  # Узел при ответе "Да"
        self.no_answer = no_answer  # Узел при ответе "Нет"

    def make_decision(self):
        """Запускаем опрос и принимаем решение на основе ответов"""
        answer = input(self.question + " (yes/no): ").strip().lower()
        if answer == "yes" and self.yes_answer:
            return self.yes_answer.make_decision()
        elif answer == "no" and self.no_answer:
            return self.no_answer.make_decision()
        else:
            return self.question  # Финальный ответ

# Создаем дерево решений
root = DecisionNode("Ты устал?")

# Ветви при "Да"
tired_yes = DecisionNode("Ты хочешь отдохнуть?")
rest_yes = DecisionNode("Отлично, сделай перерыв!")
rest_no = DecisionNode("Тогда просто выпей кофе!")
tired_yes.yes_answer = rest_yes
tired_yes.no_answer = rest_no

# Ветви при "Нет"
tired_no = DecisionNode("Ты хочешь что-то делать?")
work_yes = DecisionNode("Тогда продолжай работать!")
work_no = DecisionNode("Отлично, расслабься и отдохни!")
tired_no.yes_answer = work_yes
tired_no.no_answer = work_no

# Связываем всё в дерево
root.yes_answer = tired_yes
root.no_answer = tired_no

# Запускаем бота
print(root.make_decision())
