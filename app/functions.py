import streamlit as st
import pandas as pd
import numpy as np
import pickle

def config_page():
    st.set_page_config(page_title = "ML_AYMC", page_icon=":chart", layout = "centered")

# MOSTRAR HOME
def home():
    st.subheader("🎓 Predicción abandono o graduación universitario 📚")

    st.image("app/img/ML1.png", width="auto")

    st.markdown("""
    En este proyecto se desarrolla un **modelo de aprendizaje** capaz de predecir si un estudiante:

    - **Abandonará** sus estudios (*Dropout*) o
    - **Se graduará** al finalizar (*Graduate*)

    Para ello se utilizan **factores académicos, socioeconómicos y demográficos**.
    """)

    # Storytelling / Contexto del caso real
    with st.expander("📖 Contexto y propósito del proyecto"):
        st.markdown("""
        Eres un analísta de datos experimentado en analisis socio-académicos y el ministerio de educación de Portugal te llama para un proyecto de Machine Learning.
        
        El dilema es el siguiente, cada año se matriculan miles de estudiantes con realidades académicas, sociales y económicas muy distintas, 
        y uno de los retos más importantes para las universidades es identificar de forma temprana quiénes podrían estar en riesgo de abandonar sus estudios.

        Contar con esta información permite activar a tiempo intervenciones académicas, becas, tutorías personalizadas o apoyo psicológico, 
        aumentando de manera significativa las probabilidades de éxito del estudiante.

        Este proyecto reproduce exactamente ese escenario:
                    
        A partir de los datos que nos proporciona negocio, el objetivo es anticipar el posible desenlace académico de cada estudiante 
        y generar información útil para la toma de decisiones institucionales.

        🎯 Objetivo del modelo:

        - Detectar a mitad de curso estudiantes en riesgo, para aplicar estrategias de apoyo temprano (psicológica, académica, becas...)
                    
        - Optimizar los recursos que puede ofrecer el ministerio.

        """)

    # Información del dataset
    with st.expander("📊 Información del dataset"):
        st.markdown("""
        - **Origen:** UCI Machine Learning Repository  
        - **Instancias:** 4.424 estudiantes  
        - **Atributos:** 36  
        - **Tipo de datos:** numéricos, categóricos y socioeconómicos  
        - **Variable objetivo:**  
            - *Dropout*  
            - *Graduate*  
        - **Aplicación:** Modelos de machine learning supervisada + no supervisado
        """)

    # Pipeline del proyecto
    with st.expander("🛠️ Metodología de Machine Learning"):
        st.markdown("""
        - **Preprocesamiento:** codificación categórica, escalado, manejo del target, pipelines y Grid Cross Validation
        - **Modelos probados:**  
            - Logistic Regression  
            - Random Forest  
            - CatBoost  
            - XGBoost  
            - SVC  
        - **Clustering:** K-Means y DBSCAN para identificar perfiles  
        - **Métricas:** F1-score por clase, Matrix de confusión, ROC-AUC
        - **Interpretabilidad:** Feature Importances, RFE y SHAP para conocer los factores más influyentes  
        """)

    # Links
    with st.expander("🔗 Dataset original"):
        st.markdown("""
        🗂️ [Predict Students Dropout and Academic Success — UCI](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
        """)

# CARGA DE DATASET RAW
def carga_datos(): 
    
#   Posibilidad de que descarguen los datos de negocio
    with st.expander("📥 Descargar datos de negocio:"):
        with open("./data/1_raw/studient.csv", "rb") as f:
            st.download_button(
                label="Descargar datos de negocio",
                data=f,
                file_name="studient.csv",
                mime="text/csv"
            )
    uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=";")
        st.session_state["df"] = df
        if st.button("Ver datos"):
                st.markdown("<h3 style='text-align: center; color: white;'>DATOS DE NEGOCIO</h3>", unsafe_allow_html=True)
                st.dataframe(df)
                st.dataframe(df.shape)
                st.subheader('''*Tabla de descripción de los datos:*''')
                st.markdown('''
| DATOS | TIPO | DESCRIPCIÓN |
|-------|------|-------------|
| Marital Status | Cate. numérica | Estado civil (1: soltero, 2: casado, 3: viudo, 4: divorciado, 5: unión de hecho, 6: separación legal) |
| Application mode | Cate. numérica  | Tipo de acceso (distintas vías de admisión) |
| Application order | Cate. binario  | Orden de preferencia (0 = primera opción; 9 = última) |
| Course | Cate. numérica | Código del curso matriculado |
| Daytime/evening attendance | Cate. binario | Tipo de horario (1 = diurno, 0 = nocturno) |
| Previous qualification | Cate. numérica | Nivel educativo previo (código 1 - 43) |
| Previous qualification (grade) | Continua | Nota previa (0–200) |
| Nacionality | Cate. numérica | Código de nacionalidad del estudiante |
| Mother's qualification | Cate. numérica | Nivel educativo de la madre (código 1–44) |
| Father's qualification | Cate. numérica | Nivel educativo del padre (código 1–44) |
| Mother's occupation | Cate. numérica | Ocupación de la madre (código 0–194) |
| Father's occupation | Cate. numérica | Ocupación del padre (código 0–195) |
| Admission grade | Continua  | Nota de admisión (0–200) |
| Displaced | Cate. binario  | Vive desplazado (1 = sí, 0 = no) |
| Educational special needs | Cateo binario | Necesidades educativas especiales (1 = sí, 0 = no) |
| Debtor | Cate. binaria | Deuda con la institución (1 = sí, 0 = no) |
| Tuition fees up to date | Cate. binario | Tasas al día (1 = sí, 0 = no) |
| Gender | Cate. binario | Género (1 = hombre, 0 = mujer) |
| Scholarship holder | Cate. binario | Becado (1 = sí, 0 = no) |
| Age at enrollment | Continua | Edad al matricularse |
| International | Cate. binario | Estudiante internacional (1 = sí, 0 = no) |
| Curricular units 1st sem (credited) | Discreta | Créditos reconocidos en 1er semestre |
| Curricular units 1st sem (enrolled) | Discreta | Asignaturas inscritas en 1er semestre |
| Curricular units 1st sem (evaluations) | Discreta | Evaluaciones realizadas en 1er semestre |
| Curricular units 1st sem (approved) | Discreta | Asignaturas aprobadas en 1er semestre |
| Curricular units 1st sem (grade) | Continua | Nota media en 1er semestre (0–20) |
| Curricular units 1st sem (without evaluations) | Discreta | Asignaturas sin evaluación en 1er semestre |
| Curricular units 2nd sem (credited) | Discreta | Créditos reconocidos en 2º semestre |
| Curricular units 2nd sem (enrolled) | Discreta | Asignaturas inscritas en 2º semestre |
| Curricular units 2nd sem (evaluations) | Discreta | Evaluaciones realizadas en 2º semestre |
| Curricular units 2nd sem (approved) | Continua | Asignaturas aprobadas en 2º semestre |
| Curricular units 2nd sem (grade) | Numérico Discreta | Nota media en 2º semestre (0–20) |
| Curricular units 2nd sem (without evaluations) | Discreta | Asignaturas sin evaluación en 2º semestre |
| Unemployment rate | Continua | Tasa de desempleo (%) |
| Inflation rate | Continua | Tasa de inflación (%) |
| GDP | Continua | Producto Interno Bruto |
| Target | Categórica | Variable objetivo: *dropout*, *enrolled*, *graduate* |
                            ''')  

# COCLUSIONES DEL EDA
def coclu_eda():
    st.subheader("🔎 Conclusiones del Análisis Exploratorio de Datos 🔍")

    st.markdown("""
    ### 📊 1. Calidad del dataset
    - No hay valores nulos ni duplicados
    
    ### 🎯 2. Transformación del objetivo
    - La clase **Enrolled** fue eliminada por no representar un resultado final 🚫
    - El problema se convierte en binario: **Graduate vs Dropout**
    - Distribución resultante: **60% / 40%** ⚖️ (desbalanceo leve, a tener en cuenta)

    ### 🧭 3. Necesidades del negocio
    - Predecir el riesgo **a mitad de curso** ⏳
    - Reducir el abandono universitario 🎓
    - Optimizar recursos del Ministerio 🏛️

    ### 🧩 4. Sesgo en base a conocimiento de negocio

    **📘 Perfil académico previo**
    - *Nota de admisión* — indicador de preparación inicial.
    - *Nota media (estudios previos)* — rendimiento previo.
    - *Estudios previos* — contexto educativo del alumno.

    **👪 Contexto socio-familiar**
    - *Grado de educación (Madre)* y *Grado de educación (Padre)* — reflejan entorno educativo.
    - *¿Ha sido desplazado?* — estudiantes desplazados pueden enfrentar más dificultades.
    - *Estado civil* — situación familiar.
    - *Nacionalidad* — posibles barreras culturales o lingüísticas.

    **📚 Desempeño académico**
    - *Créditos matriculados (1re Cuatrimestre)* — asignaturas matriculadas.
    - *Créditos aprobados (1re Cuatrimestre)* — rendimiento académico .
    - *Nota media (1re Cuatrimestre)* — rendimiento académico (cuantitativo).

    **💰 Factores económicos**
    - *¿Está al día con el pago?* — (Si está al día en el pago) indicador financiero clave en riesgo de abandono.

    ### 🗂️ 5. Dataset final

    - Este dataset es la base para el preprocesamiento y modelado
    """)

# PREDICCIÓN UNITARIA
def predi_uni():
    st.title("👨‍🎓 Predicción por estudiante 👩‍🎓")

    # --------------------------
    # Carga del modelo
    # --------------------------
    with open("./models/XGBoostC_4.pkl", "rb") as entrada:
        modelo = pickle.load(entrada)

    st.write("Introduce la información del estudiante para obtener la predicción:")

    # --------------------------
    # SLIDERS
    # --------------------------

    admission_grade = st.slider(
        "Nota de admisión",
        min_value=0.0, max_value=200.0, value=100.0, step=1.0
    )

    previous_qualification_grade = st.slider(
        "Nota media (estudios previos)",
        min_value=0.0, max_value=200.0, value=100.0, step=1.0
    )

    curricular_enrolled = st.slider(
        "Créditos matriculados (1re Cuatrimestre)",
        min_value=0, max_value=30, value=6, step=6
    )

    curricular_approved = st.slider(
        "Créditos aprobados (1re Cuatrimestre)",
        min_value=0, max_value=30, value=6, step=6
    )

    curricular_grade = st.slider(
        "Nota media (1re Cuatrimestre)",
        min_value=0.0, max_value=20.0, value=10.0, step=0.5
    )

    # --------------------------
    # SELECTBOXES
    # --------------------------

    previous_qualification_options = {
        1: "Secondary education",
        2: "Higher education - bachelor's degree",
        3: "Higher education - degree",
        4: "Higher education - master's",
        5: "Higher education - doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling - not completed",
        10: "11th year of schooling - not completed",
        12: "Other - 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling - not completed",
        19: "Basic education 3rd cycle",
        38: "Basic education 2nd cycle",
        39: "Technological specialization course",
        40: "Higher education - degree (1st cycle)",
        42: "Professional higher technical course",
        43: "Higher education - master (2nd cycle)"
    }

    previous_qualification = st.selectbox(
        "Estudios previos:",
        options=list(previous_qualification_options.keys()),
        format_func=lambda x: previous_qualification_options[x]
    )

    mother_qualification_options = {
        1: "Secondary Education - 12th Year",
        2: "Higher Education - Bachelor's",
        3: "Higher Education - Degree",
        4: "Higher Education - Master's",
        5: "Higher Education - Doctorate",
        6: "Frequency of Higher Education",
        9: "12th Year - Not Completed",
        10: "11th Year - Not Completed",
        11: "7th Year (Old)",
        12: "Other - 11th Year",
        13: "2nd year complementary high school course",
        14: "10th Year",
        18: "General commerce course",
        19: "Basic Education 3rd Cycle",
        20: "Complementary High School",
        22: "Technical-professional course",
        25: "Complementary High School - not concluded",
        26: "7th year",
        27: "2nd cycle high school",
        29: "9th Year - Not Completed",
        30: "8th year",
        31: "Administration and Commerce",
        33: "Accounting and Administration",
        34: "Unknown",
        35: "Can't read or write",
        36: "Reads w/o 4th year",
        37: "Basic education 1st cycle",
        38: "Basic Education 2nd Cycle",
        39: "Technological specialization",
        40: "Higher education - degree",
        41: "Specialized higher studies",
        42: "Professional higher technical",
        43: "Higher education - master",
        44: "Higher education - doctorate"
    }

    mother_qualification = st.selectbox(
        "Grado de educación (Madre)",
        options=list(mother_qualification_options.keys()),
        format_func=lambda x: mother_qualification_options[x]
    )

    father_qualification = st.selectbox(
        "Grado de educación (Padre)",
        options=list(mother_qualification_options.keys()),
        format_func=lambda x: mother_qualification_options[x]
    )

    displaced = st.selectbox("¿Ha sido desplazado?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    marital_status_options = {
        1: "single",
        2: "married",
        3: "widower",
        4: "divorced",
        5: "facto union",
        6: "legally separated"
    }

    marital_status = st.selectbox(
        "Estado civil",
        options=list(marital_status_options.keys()),
        format_func=lambda x: marital_status_options[x]
    )

    nationality_options = {
        1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian",
        13: "Dutch", 14: "English", 17: "Lithuanian", 21: "Angolan",
        22: "Cape Verdean", 24: "Guinean", 25: "Mozambican", 26: "Santomean",
        32: "Turkish", 41: "Brazilian", 62: "Romanian",
        100: "Moldovan", 101: "Mexican", 103: "Ukrainian",
        105: "Russian", 108: "Cuban", 109: "Colombian"
    }

    nationality = st.selectbox(
        "Nacionalidad",
        options=list(nationality_options.keys()),
        format_func=lambda x: nationality_options[x]
    )

    tuition_fees = st.selectbox("¿Está al día con el pago?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # --------------------------
    # BOTÓN DE PREDICCIÓN
    # --------------------------

    if st.button("Predecir"):
        entrada = np.array([[
            admission_grade,
            previous_qualification_grade,
            previous_qualification,
            mother_qualification,
            father_qualification,
            displaced,
            marital_status,
            nationality,
            curricular_enrolled,
            curricular_approved,
            curricular_grade,
            tuition_fees
        ]])

        probas = modelo.predict_proba(entrada)[0]

        st.subheader("📌 Probabilidades por clase")
        st.write(f"Dropout: **{probas[0]:.3f}**")
        st.write(f"Graduate: **{probas[1]:.3f}**")

        pred = modelo.predict(entrada)[0]
        
        if pred == 0:
            st.success(f"Predicción final: **Dropout** ❌")
        else:
            st.success(f"Predicción final: **Graduated** 🎓")

# PREDICTOR EN FORMA DE CSV
def predi_csv():

    st.header("📊 Predicción sobre TEST: 📊")

    st.markdown("---")
    st.markdown("Carga de datos a predecir")

    # ============================================================
    # DESCARGA CSV DE EJEMPLO
    # ============================================================
    with st.expander("📥 Descargar datos de ejemplo"):
        with open("./data/4_test/X_test.csv", "rb") as f:
            st.download_button(
                label="Descargar datos",
                data=f,
                file_name="X_test.csv",
                mime="text/csv"
            )

    # ============================================================
    # SUBIR CSV DEL USUARIO
    # ============================================================
    uploaded_file1 = st.file_uploader(label="", type=["csv"])

    if uploaded_file1:
        X_test = pd.read_csv(uploaded_file1)
        
        if st.button("Ver datos cargados"):
            st.dataframe(X_test)

    st.markdown("---")
    st.markdown("Carga del target")
    
    # ============================================================
    # DESCARGA TARGET
    # ============================================================
    with st.expander("📥 Descargar target de ejemplo"):
        with open("./data/4_test/y_test.csv", "rb") as f:
            st.download_button(
                label="Descargar target",
                data=f,
                file_name="y_test.csv",
                mime="text/csv"
            )

    # ============================================================
    # SUBIR TARGET
    # ============================================================
    uploaded_file2 = st.file_uploader(label=" ", type=["csv"])

    if uploaded_file2:
        y_test = pd.read_csv(uploaded_file2)
        
        if st.button("Ver target cargado"):
            st.dataframe(y_test)

        # ============================================================
        # CARGA DEL MODELO
        # ============================================================
        with open("./models/XGBoostC_4.pkl", "rb") as entrada:
            modelo = pickle.load(entrada)

        # ============================================================
        # PREDICCIÓN
        # ============================================================
        st.markdown("---")
        if st.button("🚀 Realizar Predicciones"):
            pred = modelo.predict(X_test)
            pred_labels = pd.Series(pred).map({0: "Dropout", 1: "Graduate"})

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_test_e = le.fit_transform(y_test)

            st.subheader("📌 Predicción generada")
            st.write(pd.DataFrame({"Predicción": pred_labels}))

            from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report

            # ============================================================
            # MÉTRICAS DIRECTAMENTE CON PREDICCIÓN
            # ============================================================
            #f1_w = f1_score(y_test_e, pred, average="weighted")
            #recall_w = recall_score(y_test_e, pred, average="weighted")
            #precision_w = precision_score(y_test_e, pred, average="weighted")

            #col1, col2, col3 = st.columns(3)
            #col1.metric("🎯 F1-Weighted", f"{f1_w:.3f}")
            #col2.metric("📌 Recall", f"{recall_w:.3f}")
            #col3.metric("📐 Precision", f"{precision_w:.3f}")

            # ============================================================
            # REPORTE DEL MODELO
            # ============================================================
            #cr = classification_report(
            #    y_test_e,
            #    pred,
            #    output_dict=True,
            #    zero_division=0
            #)

            #cr_df = pd.DataFrame(cr).T.round(3)
            #cr_df = cr_df.rename(index={
            #    "0": "Dropout (0)",
            #    "1": "Graduate (1)",
            #    "macro avg": "Macro Avg",
            #    "weighted avg": "Weighted Avg"
            #})
            #st.dataframe(cr_df)

            # ============================================================
            # MATRIZ DE CONFUSIÓN
            # ============================================================
            import plotly.graph_objects as go

            conf_mat = confusion_matrix(y_test_e, pred)
            labels = ["Dropout (0)", "Graduated (1)"]

            fig = go.Figure(data=go.Heatmap(
                z=conf_mat,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=True,
                text=conf_mat,
                texttemplate="%{text}",
                textfont={"size": 14}
            ))

            fig.update_layout(
                title={
                    "text": "Matriz de Confusión",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {
                        "size": 30,     
                        "family": "Arial",
                        "color": "white"
                    }
                },
                xaxis_title="Predicción",
                yaxis_title="Real",
                width=600,
                height=600
            )
            st.plotly_chart(fig, width=True)

            # ============================================================
            # 📊 RESUMEN FINAL PARA NEGOCIO
            # ============================================================
            st.subheader("📌 Resumen final para negocio")

            TP = conf_mat[1, 1]      # Graduated reales detectados
            FN = conf_mat[0, 1]      # Dropouts NO detectados
            FP = conf_mat[1, 0]      # Graduated detectados como Dropouts
            TN = conf_mat[0, 0]      # Dropouts bien predichos

            st.markdown(f"""

            **✔️ Dropouts detectados**: {TN}  
            **⚠️ Dropouts no detectados**: {FN}  
            **🔔 Estudiantes alertados sin riesgo real**: {FP}  
            **🎓 Graduados correctos**: {TP}
            """)

            # ============================================================
            # 📊 CÁLCULO ROI DEL MODELO
            # ============================================================
            #Aquí podría haber metido un slide para el theshold

            st.markdown("---")

            with st.expander("Mostar el ROI💰"):

                st.subheader("💰 Impacto Económico — ROI")

                # Variables de negocio (Una estimación)
                VALOR_RENDIMIENTO = 7000  # € alumno salvado
                COSTE_INTERVENCION = 3500  # € intervención alumno
                COSTE_MODELO = 5000       # coste total del proyecto ML

                a_salvado, c_estudiante, c_modelo = st.columns(3)
                a_salvado.metric("✅ Beneficio por alumno ayudado", f"{VALOR_RENDIMIENTO} €")
                c_estudiante.metric("❌ Pérdida por alumno NO detectado", f"-{COSTE_INTERVENCION} €")
                c_modelo.metric("⚙️ Coste del modelo", f"{COSTE_MODELO} €")

                # Cálculos
                beneficio = TN * VALOR_RENDIMIENTO
                costo_fp = FP * COSTE_INTERVENCION
                costo_tp = TN * COSTE_INTERVENCION
                perdida_fn = FN * VALOR_RENDIMIENTO

                beneficio_neto = beneficio - costo_fp - costo_tp - perdida_fn
                roi = (beneficio_neto - COSTE_MODELO) / COSTE_MODELO


                st.markdown(f"""
                ### Interpretación 💡

                - **Alumnos ayudados**: `{TN}` → generan ingresos → `{beneficio:,.0f} €`
                - **Alumnos perdidos**: `{FN}` → pérdidas directas → `{perdida_fn:,.0f} €`
                - **Intervenciones realizadas** a estudiantes en riesgo → `{TN + FP}`
                - **Coste operativo total** → `{costo_tp + costo_fp:,.0f} €`

                ---
            
                """)

                colA, colB, colC = st.columns(3)
                colA.metric("💸 Beneficio por retenciones", f"{beneficio:,.0f} €")
                colB.metric("📉 Pérdida por NO detectados", f"-{perdida_fn:,.0f} €")
                colC.metric("🎯 ROI final", f"{roi*100:.1f} %")

                st.markdown(f"""

                ---
                ## **Beneficio neto del modelo** → `{beneficio_neto:,.0f} €`
                """)

