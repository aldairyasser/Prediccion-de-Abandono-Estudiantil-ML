# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.metrics import roc_curve, roc_auc_score

import pickle 

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LEEMOS LOS DATOS PARA EL MODELO
# ============================================================

X_test = pd.read_csv("../data/4_test/X_test.csv")
y_test_e = pd.read_csv("../data/4_test/y_test_XGB.csv")
y_test = pd.read_csv("../data/4_test/y_test.csv")

# ============================================================
# LEEMOS EL MODELO FINAL
# ============================================================

path = "../models/XGBoostC_4.pkl"
with open(path, 'rb') as archivo_entrada:
    modelo_xgb_importado = pickle.load(archivo_entrada)

# modelo_xgb_importado.predict_proba(X_test)

# ============================================================
# MÉTRICAS DE NEGOCIO
# ============================================================

# Predicciones
y_pred = modelo_xgb_importado.predict(X_test)

# Métricas
print("\nAccuracy:", accuracy_score(y_test_e, y_pred))
print("\nPrecision:", precision_score(y_test_e, y_pred, average="weighted"))
print("\nRecall:", recall_score(y_test_e, y_pred, average="weighted"))
print("\nF1-score:", f1_score(y_test_e, y_pred, average="weighted"))
print("\nReporte de la clasificación:\n", classification_report(y_test_e, y_pred))

# Matriz de confusión
conf_mat = confusion_matrix(y_test_e, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Confusion Matrix - XGBoost Classifier 4ta Vuelta")
plt.show()

# A negocio le interesa el precision/recall

# ============================================================
# CURVA AUC-ROC
# ============================================================

y_pred = modelo_xgb_importado.predict_proba(X_test)[:, 1] # Graduate

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label="Graduate") # 0.919

# AUC
roc_auc_xgb = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_xgb:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - XGBoost")
plt.legend()
plt.show()
