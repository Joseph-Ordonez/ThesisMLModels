import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

# Dataset
data = pd.read_csv('/content/ShortSyntheticData.csv', sep=";")

# Codificacion de valores en la columna "Career"
label_encoder = LabelEncoder()
data['Career'] = label_encoder.fit_transform(data['Career'])

# Variables independientes y variable dependiente
X = data.drop(columns=['Career'])  # Independientes
y = data['Career']  # Dependiente

# Uso de SMOTE para el balanceo de clases
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Tecnica de Stratified Shuffle Split
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

# Seleccion de caracteristicas con RFE
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
selector = RFE(estimator=rf_model, n_features_to_select=30, step=1)
selector = selector.fit(X_train, y_train)

# Aplicación de la selección de características en los conjuntos de entrenamiento y prueba
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Ajuste de hiperparámetros con GridSearchCV
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_selected, y_train)

# Entrenamiento del modelo con los mejores parámetros encontrados
best_rf_model = grid_search.best_estimator_

# Realizar predicciones con el conjunto pruebas
y_pred = best_rf_model.predict(X_test_selected)

# Evaluación de rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Características
selected_features = X.columns[selector.support_]
feature_importances = best_rf_model.feature_importances_
for feature, importance in zip(selected_features, feature_importances):
    print(f"Feature: {feature}, Importance: {importance}")
