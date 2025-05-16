# Proyecto: Predicción de Ganador de Partidos de la Liga Peruana de Fútbol (2014-2024)

## 1. Motivo del proyecto

Este proyecto nace de la combinación de mi pasión por el fútbol y mi interés en la ciencia de datos. El objetivo principal es adquirir y perfeccionar habilidades en:

* **Web scraping**: Extraer datos históricos de partidos de la Liga peruana (2014-2024) desde fbref.com.
* **Procesamiento y limpieza** de datos: Garantizar la calidad y consistencia del dataset.
* **Ingeniería de características**: Diseñar y seleccionar las variables más relevantes para la predicción.
* **Modelado predictivo**: Implementar y comparar distintos algoritmos de machine learning para anticipar el ganador (equipo local o visitante).

Además, como resultado de este trabajo, se creó y publicó un dataset en Kaggle, disponible en: [Matches of the Peruvian Soccer League 2014-2024](https://www.kaggle.com/datasets/ralcasanova/matches-of-the-peruvian-soccer-league-2014-2024).

## 2. Resultados del proyecto

* **Dataset personalizado**: Más de 10 años de estadísticas de la Liga 1 peruana, almacenadas en `data/Liga_1_Matches_2014-2024.csv`.
* **Notebook de análisis**: `notebook/Predict_match_winner.ipynb`, con visualizaciones y comparación de modelos.
* **Notebook oficial en Kaggle**: [Will the Home Team Win?](https://www.kaggle.com/code/ralcasanova/will-the-home-team-win), donde se presenta el análisis completo del proyecto, incluyendo entrenamiento, evaluación y conclusiones.
* **Modelos entrenados**:

  * **Logistic Regression**
  * **Decision Tree Classifier**
  * **Random Forest Classifier**
  * **Support Vector Machine**
  * **XGBoost Classifier**

### Resultados destacados

**Mejor estimador con todos los features: LogisticRegression**

```
              precision    recall  f1-score   support

           0       0.54      0.59      0.57       333
           1       0.58      0.53      0.55       352

    accuracy                           0.56       685
   macro avg       0.56      0.56      0.56       685
weighted avg       0.56      0.56      0.56       685
```

**Mejor estimador con feature selector: LogisticRegression**

```
              precision    recall  f1-score   support

           0       0.57      0.64      0.60       333
           1       0.61      0.54      0.57       352

    accuracy                           0.59       685
   macro avg       0.59      0.59      0.59       685
weighted avg       0.59      0.59      0.59       685
```

## 3. Estructura del proyecto

```text
 .
├──  __init__.py
├──  classify.py
├──  data
│   └──  Liga_1_Matches_2014-2024.csv
├──  data_cleaning.py
├──  fbref_cache.sqlite
├──  feature_engineer.py
├──  main.py
├──  notebook
│   └──  Predict_match_winner.ipynb
├── 󰂺 README.md
├──  requirements.txt
├──  utils.py
└──  web_scrapping
    ├──  __init__.py
    ├──  main.py
    └──  url_teams.csv
```

## 4. Instalar y ejecutar localmente

Sigue estos pasos para ejecutar el proyecto en tu máquina local (Linux):

1. **Clonar el repositorio**:

   ```bash
   git clone <URL-del-repositorio>
   cd <nombre-del-proyecto>
   ```

2. **Crear un entorno virtual**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Ejecutar el pipeline completo**:

   ```bash
   python main.py --scrape   # Extrae datos de fbref.com
   python main.py --train    # Entrena y evalúa los modelos
   ```

5. **Analizar resultados**:
   Abre el notebook con:

   ```bash
   jupyter notebook notebook/Predict_match_winner.ipynb
   ```

## 5. Contacto

**Raúl Casanova**

* ✉️ Email: [raul.casanova.03@gmail.com](mailto:raul.casanova.03@gmail.com)
* 🔗 LinkedIn: [raul-casanova](https://www.linkedin.com/in/raul03-casanova28/)
* 📊 Kaggle: [ralcasanova](https://www.kaggle.com/ralcasanova)

> Si estás interesado en brindar oportunidades laborales relacionadas con ciencia de datos, no dudes en escribirme. Estaré encantado de conversar.

---

*Desarrollado con pasión por el fútbol y la ciencia de datos.*

