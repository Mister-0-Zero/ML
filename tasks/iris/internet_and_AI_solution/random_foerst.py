from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def random_forest(X_train, X_test, y_train, y_test):
    # Создаем модель Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Обучаем модель
    rf_model.fit(X_train, y_train)

    # Делаем прогнозы на тестовых данных
    y_pred = rf_model.predict(X_test)

    return accuracy_score(y_test, y_pred)