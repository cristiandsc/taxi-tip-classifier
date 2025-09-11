from pathlib import Path

readme_text = """# 🚖 Taxi Tip Classifier

Proyecto de machine learning para predecir si un viaje de taxi en NYC tendrá una propina **alta** (>20% de la tarifa).  
Se utiliza un flujo modular basado en notebooks y scripts en `src/` siguiendo buenas prácticas de MLOps.

---

## 📂 Estructura del proyecto

```bash
taxi-tip-classifier/
├── data/                # Datos locales (no versionados en Git)
│   ├── raw/             # Descargas originales
│   ├── interim/         # Datos intermedios
│   └── processed/       # Datos listos para modelar (ej: taxi_train.parquet)
│
├── models/              # Modelos entrenados (.joblib)
│
├── notebooks/           # Notebooks exploratorios y pipeline
│   ├── 00_nyc_taxi_model.ipynb
│   ├── 01-cvasquezp-descarga-data.ipynb
│   ├── 02-cvasquezp-preprocesa-data.ipynb
│   ├── 03-cvasquezp-entrena-modelo.ipynb
│   └── 04-cvasquezp-evalua-modelo.ipynb
│
├── reports/             # Reportes y métricas
│   └── metrics_eval_feb2020.json
│
├── src/                 # Código modular
│   ├── config.py
│   ├── data/
│   │   └── dataset.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       ├── train_model.py
│       └── predict_model.py
│
├── requirements.txt     # Dependencias con versiones
├── LICENSE
└── README.md

⚙️ Instalación

Clona este repositorio:

git clone https://github.com/<usuario>/taxi-tip-classifier.git
cd taxi-tip-classifier


Crea un entorno virtual y activa:

conda create -n taxi python=3.10 -y
conda activate taxi


Instala dependencias:

pip install -r requirements.txt

🚀 Flujo de trabajo
1. Descarga de datos

Ejecutar notebooks/01-cvasquezp-descarga-data.ipynb

Descarga el dataset de enero 2020 (NYC Taxi).

Guarda datos crudos en data/raw/.

2. Preprocesamiento

Ejecutar notebooks/02-cvasquezp-preprocesa-data.ipynb

Crea la variable objetivo high_tip (>20%).

Genera features (trip_time, trip_speed, horarios, etc.).

Guarda dataset preprocesado en data/processed/taxi_train.parquet.

3. Entrenamiento

Ejecutar notebooks/03-cvasquezp-entrena-modelo.ipynb

Entrena un RandomForestClassifier.

Calcula métrica F1 en train.

Guarda el modelo en models/random_forest.joblib.

4. Evaluación

Ejecutar notebooks/04-cvasquezp-evalua-modelo.ipynb

Carga datos de febrero 2020.

Evalúa el modelo guardado.

Genera métricas de validación (reports/metrics_eval_feb2020.json).

📊 Resultados

Ejemplo de métricas en febrero 2020:

precision    recall  f1-score   support
0    0.688     0.130     0.219   2676852
1    0.596     0.956     0.735   3600002


El modelo identifica bien los viajes con propina alta (recall=0.95).

A mejorar: balance entre precisión y recall para clases negativas.

🛠️ Tecnologías usadas

Python 3.10

pandas, numpy

scikit-learn

pyarrow

joblib

Jupyter Notebooks

📌 Notas

Los datasets completos de NYC Taxi son públicos y están en formato Parquet.

Los archivos dentro de data/ no se versionan en GitHub (solo .gitkeep).

Este repo está organizado siguiendo prácticas de cookiecutter data science.

✍️ Autor: Cristian Vásquez P.
📅 Año: 2025
📜 Licencia: MIT
