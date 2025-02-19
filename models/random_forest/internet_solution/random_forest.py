# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Загружаем набор данных Iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Признаки
y = pd.Series(data.target)  # Целевой признак (классы)

# Разделяем данные на обучающую и тестовую выборки (40% - обучение, 6a0% - тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Создаем модель Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучаем модель
rf_model.fit(X_train, y_train)

# Делаем прогнозы на тестовых данных
y_pred = rf_model.predict(X_test)

# Оцениваем модель
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Печатаем подробный отчёт о классификации
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
