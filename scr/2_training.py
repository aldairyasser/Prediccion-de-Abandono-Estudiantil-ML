# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesado
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

# Modelos supervisados
from xgboost import XGBClassifier

# Métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.metrics import roc_curve, roc_auc_score

import pickle 

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LEEMOS LOS DATOS
# ============================================================

df = pd.read_csv("../data/2_processed/datos_limpios.csv")

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

# Dividimos en X e y
X = df.drop("Target", axis=1)
y = df["Target"]

# Separamon en X_train, X_test, y_train, y_test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ============================================================
# GUARDAMOS LOS DATOS CON EL SPLIT EN CSV's
# ============================================================

X_train.to_csv("../data/3_train/X_train.csv", index=False)
y_train.to_csv("../data/3_train/y_train.csv", index=False)
X_test.to_csv("../data/4_test/X_test.csv", index=False)
y_test.to_csv("../data/4_test/y_test.csv", index=False)

# ============================================================
#  ENCODING PARA XGBOOST
# ============================================================

le = LabelEncoder()
y_train_e = le.fit_transform(y_train)
y_test_e = le.transform(y_test)

# Guardamos la y codificada para el split de XGBoost

pd.DataFrame({"y_train_encoded": y_train_e}).to_csv("../data/3_train/y_train_XGB.csv", index=False)
pd.DataFrame({"y_test_encoded": y_test_e}).to_csv("../data/4_test/y_test_XGB.csv", index=False)

# ============================================================
# ENTRENAMIENTO DEL MODELO
# ============================================================

pipe_xgb = Pipeline([
    ("class", XGBClassifier(
        tree_method="hist",
        use_label_encoder=True 
    ))
])

param_grid_xgb = {
    "class__n_estimators": [350, 400],
    "class__learning_rate": [0.01, 0.02],  
    "class__max_depth": [4, 5, 6],
    "class__subsample": [0.85, 0.9],
    "class__colsample_bytree": [0.7, 0.8, 0.9],
    "class__min_child_weight": [2, 3],
    "class__gamma": [0.3, 0.4],
    "class__reg_alpha": [0, 0.01],
    "class__reg_lambda": [1, 2],

    "class__max_bin": [560],
    "class__max_leaves": [None, 60] 
}

grid_xgb = GridSearchCV(
    estimator=pipe_xgb,
    param_grid=param_grid_xgb,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train_e)

# ============================================================
# GUARDAMOS EL MODELO FINAL
# ============================================================

# usa pickle y guarda el estimador GridsearchCV en un archivo
import pickle

filename = "../models/XGBoostC_4.pkl"

modelo_xgb_4 = grid_xgb.best_estimator_

with open(filename, "wb") as archivo_salida:
    pickle.dump(modelo_xgb_4, archivo_salida)