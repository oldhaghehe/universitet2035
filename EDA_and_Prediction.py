

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap

# Импорт функций из наших модулей
from data_preprocessing import load_data, preprocess_data
from train_model import build_mlp_model, compile_model, train_model, evaluate_model, save_model

# # Проект по прогнозированию оттока клиентов для «Телеком-Плюс»

# ## 1. Введение
# Этот скрипт демонстрирует полный пайплайн решения задачи прогнозирования оттока клиентов для компании «Телеком-Плюс».
# Проект включает в себя разведочный анализ данных (EDA), feature engineering, построение автоматизированного пайплайна предобработки данных,
# разработку и обучение нейронной сети (MLP) с использованием TensorFlow/Keras, а также интерпретацию модели с помощью SHAP и анализ ошибок.

# ## 2. Настройка окружения и импорт библиотек
# Убедитесь, что у вас установлены все необходимые библиотеки. Если нет, раскомментируйте и выполните следующие команды:
# # pip install pandas numpy scikit-learn imbalanced-learn tensorflow shap matplotlib seaborn


# ## 3. Загрузка данных
# Загрузим данные об оттоке клиентов с помощью функции `load_data` из `data_preprocessing.py`.

file_path = "churn-bigml-80.csv" # Убедитесь, что файл находится в той же директории, что и скрипт
df = load_data(file_path)

if df is None:
    print("Ошибка загрузки данных. Проверьте путь к файлу.")
else:
    print("\nПервые 5 строк данных:")
    print(df.head())

# ## 4. Разведочный анализ данных (EDA)
# Проведем первичный анализ данных, чтобы понять их структуру, распределение и выявить потенциальные проблемы.

# ### 4.1 Общая информация о данных

print("\n--- Общая информация о DataFrame ---")
df.info()

print("\n--- Пропущенные значения ---")
print(df.isnull().sum())

print("\n--- Описательная статистика ---")
print(df.describe())

# ### 4.2 Анализ категориальных признаков

print("\n--- Уникальные значения для категориальных столбцов ---")
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

for col in categorical_cols:
    print(f"\nКолонка '{col}':")
    print(df[col].value_counts())
    if col == 'Churn':
        sns.countplot(x=col, data=df)
        plt.title(f'Распределение {col}')
        plt.show()
    elif col in ['International plan', 'Voice mail plan']:
        sns.countplot(x=col, hue='Churn', data=df)
        plt.title(f'Распределение {col} по Churn')
        plt.show()

# **Выводы EDA:**
# *   [Здесь будут конкретные выводы после выполнения ячеек выше. Например: "Отсутствуют явные пропущенные значения.", "Признак 'State' имеет 51 уникальное значение.", "Наблюдается сильный дисбаланс классов в целевой переменной 'Churn' (около 85% False, 15% True).", "Клиенты с международным планом чаще уходят." и т.д.]

# ## 5. Предобработка данных
# Используем функцию `preprocess_data` из `data_preprocessing.py` для автоматизированной предобработки данных,
# включая разделение на обучающую и тестовую выборки, масштабирование, кодирование и балансировку классов с помощью SMOTE.

X_train_resampled, X_test_processed, y_train_resampled, y_test, preprocessor_pipeline, numerical_features, categorical_features = preprocess_data(df.copy())

# Получим имена признаков после OneHotEncoding для SHAP
ohe_feature_names = preprocessor_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

print("\nРазмерность данных после предобработки:")
print(f"X_train_resampled: {X_train_resampled.shape}")
print(f"y_train_resampled: {y_train_resampled.shape}")
print(f"X_test_processed: {X_test_processed.shape}")
print(f"y_test: {y_test.shape}")

# ## 6. Построение и обучение модели глубокого обучения (MLP)
# Используем функции из `train_model.py` для создания, компиляции и обучения нейронной сети.

# Определяем входную размерность для модели
input_shape = X_train_resampled.shape[1]

# Строим модель
model = build_mlp_model(input_shape)
model.summary()

# Компилируем модель
model = compile_model(model)

# Обучаем модель
print("\n--- Обучение модели ---")
history = train_model(model, X_train_resampled, y_train_resampled, X_test_processed, y_test, epochs=100, batch_size=64)

# Визуализация истории обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('История обучения модели')
plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('История AUC')
plt.xlabel('Эпоха')
plt.ylabel('AUC')
plt.legend()
plt.show()

# ## 7. Оценка модели
# Оценим производительность обученной модели на тестовом наборе данных.

accuracy, precision, recall, f1, roc_auc, cm = evaluate_model(model, X_test_processed, y_test)

# Визуализация матрицы ошибок
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
disp.plot(cmap=plt.cm.Blues)
plt.title('Матрица ошибок')
plt.show()

# ## 8. Интерпретация модели с помощью SHAP
# Применим SHAP (SHapley Additive exPlanations) для объяснения предсказаний модели на индивидуальном и глобальном уровне.

# SHAP требует модель, которая принимает numpy массивы
# Keras модель уже работает с numpy, но для SHAP Explainer может потребоваться обертка
# Создадим функцию-обертку для предсказаний модели
def model_predict(data):
    return model.predict(data).flatten()

# Создаем SHAP explainer
# Для Keras моделей можно использовать shap.DeepExplainer или shap.KernelExplainer
# DeepExplainer быстрее, но требует специфической архитектуры (например, без сложных кастомных слоев)
# KernelExplainer более универсален, но медленнее
# Попробуем DeepExplainer
try:
    explainer = shap.DeepExplainer(model, X_train_resampled[:100]) # Используем часть обучающей выборки для background data
    shap_values = explainer.shap_values(X_test_processed)
    print("Используется DeepExplainer.")
except Exception as e:
    print(f"DeepExplainer не сработал: {e}. Попытка использовать KernelExplainer (может быть медленным).")
    explainer = shap.KernelExplainer(model_predict, X_train_resampled[:50]) # Меньше background data для скорости
    shap_values = explainer.shap_values(X_test_processed)
    print("Используется KernelExplainer.")

# Глобальная интерпретация: Summary Plot
# shap_values для бинарной классификации возвращает список из двух массивов (для класса 0 и класса 1)
# Нам нужен shap_values для класса 1 (отток)
if isinstance(shap_values, list):
    shap_values_for_churn = shap_values[0] # Для бинарной классификации часто берется первый элемент
else:
    shap_values_for_churn = shap_values

shap.summary_plot(shap_values_for_churn, X_test_processed, feature_names=all_feature_names)
plt.show() # Добавлено для отображения графика

# Локальная интерпретация: Force Plot для отдельного предсказания
# Выберем случайный пример из тестовой выборки
sample_idx = np.random.randint(0, X_test_processed.shape[0])
print(f"\nSHAP Force Plot для примера {sample_idx}:")
# shap.initjs() # Удалено, так как это для Jupyter
# shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
#                 shap_values_for_churn[sample_idx,:], 
#                 X_test_processed[sample_idx,:], 
#                 feature_names=all_feature_names)
# Force plot не отображается напрямую в скрипте, его нужно сохранять или использовать в интерактивной среде.
# Для скрипта можно использовать shap.waterfall_plot или shap.decision_plot

# Пример waterfall plot для скрипта
shap.waterfall_plot(shap.Explanation(values=shap_values_for_churn[sample_idx], 
                                    base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                    data=X_test_processed[sample_idx], 
                                    feature_names=all_feature_names))
plt.show() # Добавлено для отображения графика

# **Выводы SHAP:**
# *   [Здесь будут выводы по важности признаков и их влиянию на предсказания. Например: "Наиболее важными признаками, влияющими на отток, являются 'Total day charge', 'Customer service calls' и 'International plan_Yes'.", "Высокие значения 'Total day charge' увеличивают вероятность оттока."]

# ## 9. Анализ ошибок модели
# Проанализируем, на каких примерах модель чаще всего ошибается.

y_pred_proba = model.predict(X_test_processed).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Создадим DataFrame для анализа ошибок
error_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
error_df['Actual'] = y_test.values
error_df['Predicted'] = y_pred
error_df['Predicted_proba'] = y_pred_proba
error_df['Error'] = error_df['Actual'] != error_df['Predicted']

# False Positives (модель предсказала отток, но клиент не ушел)
fp_df = error_df[(error_df['Actual'] == 0) & (error_df['Predicted'] == 1)]
print("\n--- False Positives (FP) ---")
print(f"Количество FP: {len(fp_df)}")
if not fp_df.empty:
    print("Примеры FP (первые 5):")
    print(fp_df.head())
    # Можно провести дополнительный анализ признаков для FP

# False Negatives (модель предсказала, что клиент не уйдет, но он ушел)
fn_df = error_df[(error_df['Actual'] == 1) & (error_df['Predicted'] == 0)]
print("\n--- False Negatives (FN) ---")
print(f"Количество FN: {len(fn_df)}")
if not fn_df.empty:
    print("Примеры FN (первые 5):")
    print(fn_df.head())
    # Можно провести дополнительный анализ признаков для FN

# **Выводы по анализу ошибок:**
# [Например: "Модель чаще ошибается, предсказывая отток там, где его нет (FP), что может быть связано с агрессивной стратегией SMOTE.",
# "FN часто имеют средние значения признаков, что делает их сложными для классификации."]

# ## 10. Бизнес-рекомендации

# На основе полученной модели, SHAP-анализа и анализа ошибок, предложим бизнес-рекомендации.

# **Пример рекомендаций:**
# *   **Для клиентов с высоким значением SHAP-фактора 'Total day charge' (высокая дневная плата):** Предлагать персональные скидки или более выгодные тарифы, чтобы снизить финансовую нагрузку и предотвратить отток.
# *   **Для клиентов с высоким количеством 'Customer service calls' (много звонков в службу поддержки):** Улучшить качество обслуживания, возможно, назначить персонального менеджера или предложить более быстрые каналы решения проблем. Это указывает на неудовлетворенность сервисом.
# *   **Для клиентов с 'International plan_Yes' (международный план):** Разработать специальные предложения или бонусы, ориентированные на их потребности, так как они демонстрируют повышенный риск оттока.
# *   **Общие рекомендации:**
#     *   Регулярно мониторить ключевые признаки, выявленные SHAP, для раннего выявления потенциально уходящих клиентов.
#     *   Разработать проактивные кампании удержания, основанные на индивидуальных профилях риска.

# ## 11. Сохранение модели

# Сохраним обученную модель для дальнейшего использования.

model_save_path = "churn_mlp_model.h5"
save_model(model, model_save_path)

