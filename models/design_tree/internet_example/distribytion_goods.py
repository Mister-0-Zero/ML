class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child_node):
        """Добавляет подкатегорию (ребёнка) к текущей категории"""
        self.children.append(child_node)

    def display(self, level=0):
        """Выводит дерево в виде структуры"""
        print("  " * level + f"- {self.name}")
        for child in self.children:
            child.display(level + 1)

# Создаем корневой узел
root = TreeNode("Товары")

# Создаем категории
electronics = TreeNode("Электроника")
clothing = TreeNode("Одежда")

# Добавляем подкатегории
laptops = TreeNode("Ноутбуки")
phones = TreeNode("Смартфоны")
electronics.add_child(laptops)
electronics.add_child(phones)

tshirts = TreeNode("Футболки")
jeans = TreeNode("Джинсы")
clothing.add_child(tshirts)
clothing.add_child(jeans)

# Добавляем категории в корень
root.add_child(electronics)
root.add_child(clothing)

# Выводим дерево
root.display()