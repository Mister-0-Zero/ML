import numpy as np


# Узел дерева
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Индекс признака для разбиения
        self.threshold = threshold  # Порог для разбиения
        self.left = left  # Левое поддерево
        self.right = right  # Правое поддерево
        self.value = value  # Класс в листе (если узел - лист)

    def is_leaf(self):
        return self.value is not None


# Класс дерева решений
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Строим дерево решений"""
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """Рекурсивное построение дерева"""
        num_samples, num_features = X.shape

        # Условие остановки (если глубина достигнута или слишком мало элементов)
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(set(y)) == 1:
            return Node(value=self._most_common_label(y))

        # Ищем лучшее разбиение
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # Разбиваем данные
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y):
        """Находит наилучший признак и порог для разбиения"""
        num_samples, num_features = X.shape
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._gini_index(X[:, feature], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, feature_column, y, threshold):
        """Вычисляет Gini impurity (нечистоту Джини)"""
        left_mask = feature_column <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 1  # Максимальная нечистота, если разбиение невозможно

        def gini(y_subset):
            """Формула критерия Джини"""
            classes, counts = np.unique(y_subset, return_counts=True)
            probs = counts / len(y_subset)
            return 1 - np.sum(probs ** 2)

        left_gini = gini(y[left_mask])
        right_gini = gini(y[right_mask])

        left_weight = np.sum(left_mask) / len(y)
        right_weight = np.sum(right_mask) / len(y)

        return left_weight * left_gini + right_weight * right_gini

    def _most_common_label(self, y):
        """Возвращает наиболее частую метку класса"""
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X):
        """Предсказываем класс для каждого объекта"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Проход по дереву для предсказания"""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# import numpy as np
#
# class MyDicisionTreeClassifier:
#     def init(self, max_depth = 5, kol_data_in_classification = 10):
#         self.max_depth = max_depth
#         self.kol_data_in_classification = kol_data_in_classification
#         self.head = self.Node()
#
#     def Node(self, value = None, next_left = None, next_right = None):
#         self.value = value
#         self.next_left = next_left
#         self.next_right = next_right
#
#     def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
#         self.X_train = X_train
#         self.Y_train = Y_train
#         self.X_train_T = X_train.T
#         self.Y_train_T = Y_train.T
#         self.fiting(depth = 0)
#
#     def fiting(self, depth):
#         pass
#
#     def best_feature(self):
#         n_samples, n_features = self.X_train.shape
#         coef_gini = float("inf")
#         best_feature = -1
#
#         for feature in range(n_features):
#             coef_gini_current = gini(self.X_train_T[feature])
#             if coef_gini_current and coef_gini_current < coef_gin:
#                 coef_gini = coef_gini_current
#                 best_feature = feature
#
#         if best_feature != -1:
#             return best_feature
#         else:
#             pass
#
#     def gini(self, X):
#             pass