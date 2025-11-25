# 🧠 Predicción de Abandono Estudiantil — ML
Este proyecto busca predecir si un estudiante universitario abandonará o se graduará, utilizando un dataset real de estudiantes.

El problema tiene impacto directo en retención académica, eficiencia de recursos y bienestar estudiantil.

---

## 📌 Descripción del proyecto

El abandono universitario es una de las principales preocupaciones de instituciones educativas, ya que genera:

Pérdida de matrícula

Desmotivación y desgaste emocional

Desigualdad en resultados

Impacto negativo en reputación académica

Objetivo:

Construir un modelo que identifique alumnos en riesgo de Dropout para aplicar intervención temprana.

---

## 📊 Dataset

📦 **Fuente:** UCI Machine Learning Repository  
🔗 **Dataset:** [Predict students dropout and academic success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

| Característica | Valor |
|---|---|
| Instancias | ~4.400 estudiantes |
| Variables | 36 |
| Tipos de datos | Demográficos, Académicos, Económicos e Historial de calificaciones |
| Target | Dropout (Abandono), Graduate (Termina los estudios), Enrolled (eliminado del análisis) |

---

## 🧠 Modelos empleados

Preprocesamiento

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

Se implementaron **5 modelos supervisados y 1 no supervisado**:

### 📌 Supervisados (Clasificación)
- Logistic Regression (baseline + optimizado)
- Random Forest Classifier
- XGBoost
- CatBoost
- Support Vector Classifier (SVC)

Cada modelo incluye:
- Pipeline
- GridSearchCV
- Métricas de validación

Validación: CV=5
Métrica de optimización: f1_weighted

### 📌 No supervisado
**K-Means → Clusterización de perfiles estudiantiles**

Se evaluó:
- K-óptimo
- Score de silueta
- Interpretación de clusters

---

## 📈 Métricas aplicadas

Debido al desbalance de clases:

- **Accuracy → descartada**
- **Precision (Graduated)** → minimizar falsos positivos
- **Recall (Dropout)** → no dejar escapar dropouts
- **F1-weighted → métrica principal**

> El F1-weighted pondera el resultado en función del soporte de cada clase y mejora la objetividad en datasets desbalanceados.

---

## 🔧 Optimización de modelos

- Búsqueda de hiperparámetros mediante **GridSearchCV**
- Selección de modelo por rendimiento en test
- Ajuste de threshold para equilibrar:
  - Minimización de FP
  - Captura de Dropouts

---

## 🔍 Interpretabilidad (Explainability)

El modelo final se analizó mediante **SHAP (SHapley Additive Explanations):**

- Identificación de variables con mayor impacto
- Explicaciones globales
- Explicaciones para casos individuales

> La explicabilidad es clave para justificar decisiones ante equipos pedagógicos.

---

## ⚠️ Riesgos y limitaciones

- Dataset de una única institución

- Variables no incluyen motivación o psicología

- No se modela evolución temporal del estudiante

- Riesgo de sesgo demográfico

## 🧪 Instrucciones de ejecución

📦 1. Clonar repositorio

```
git clone https://github.com/usuario/proyecto-dropout-ML.git
```

🐍 2. Crear entorno

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

▶️ 3. Ejecutar Streamlit

Moverte al directorio de la app

```
cd app_streamlit
```

Ejecutar el Front

```
streamlit run app.py
```

---

## 📂 Estructura del repositorio
```
|-- data
|   |-- raw              <- dataset original
|   |-- processed        <- dataset transformado
|   |-- train            <- splitting
|   |-- test             <- splitting
|
|-- notebooks
|   |-- 01_Fuentes.ipynb
|   |-- 02_LimpiezaEDA.ipynb
|   |-- 03_Entrenamiento.ipynb
|
|-- src
|   |-- 1_data_processing.py
|   |-- 2_training.py
|   |-- 3_evaluation.py
|
|-- models
|   |-- otros
|   |-- final_model.pkl
|
|-- app_streamlit
|   |-- img
|   |-- app.py
|   |-- funtions.py
|   |-- requirements.txt
|
|-- docs
|   |-- negocio.ppt
|   |-- ds.ppt
|   |-- memoria.md
|
|-- README.md
```

---

## 🧭 Conclusión

- El abandono es predecible con alta fiabilidad combinando datos académicos y administrativos.
- Los modelos basados en árboles superan a modelos lineales para este problema.
- El ajuste de threshold permite controlar falsos positivos y proteger a estudiantes en riesgo.
- SHAP facilita la comunicación con stakeholders no técnicos.

---

## 👤 Autor

Aldair Yasser Meza Carrasco
Bootcamp Data Science — Machine Learning Project