# importar liberias necesarias
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Título 
st.title("Screening of Autism")

# Añade diferentes widgets y funcionalidades
st.write("Desarrollemos modelos para predecir si un niño tiene rasgos del espectro autista")
imagen_ti = st.image('img/Autism_simbol.png')


# Menú en el sidebar
st.sidebar.title("Menú")
opcion = st.sidebar.selectbox(
    "Elige una opción",
    ("Inicio", "EDA", "Modelo de Predicción", "Predicción en vivo")
)


# Usar la opción seleccionada para mostrar contenido
if opcion == "Inicio":
    st.write("Comencemos, en la barra de la izquierda podrás elegir que quieres ver primero")
elif opcion == "EDA":
    st.header("EDA")
    st.subheader("Entendamos los datos. \n") #Me gustaría que fuera un subtitulo
    st.write("El desafio fue plantear la posibilidad de desarrollar un modelo que a través de los pixeles de las imagenes pudiera predecir la clase de los niños, haciendo referencia a si tienen rasgos del espectro autista")
    sample = st.image('img/sample.png')
    st.write('Acá veremos una visualización de un sample sacado de nuestro conjunto de datos, para esta etapa del problema las imagenes estaban codificadas en grises para optimizar el proceso de aprendizaje de los modelos')
    st.write('\n')
    st.subheader('Ahora veamos como se distribuye la clase') #Me gustaría que fuera un subtitulo
    distri_clase = st.image('img/distribucion_clases.png')
    distri_clase_train_test = st.image('img/distribucion_clase_train_test.png')
    st.write('Decidimos mantener la cantidad de pixeles de origen, 224x224, que aunque para el procesado del mismo sería un proceso computacional denso, utilizamos un enfoque de PCA para disminuir las dimensiones luego de que aplanaramos las imagenes, primer paso para desarrolar los modelos de sklearn \n')
    with st.expander('Análisis   PCA'):
        st.write('DF de los PCA')
        df_pca = pd.read_csv('data/df_pca.csv')
        st.write(df_pca)
        st.write('Porcentaje de Varianza Explicada Acumulada')
        varianza_explicada = st.image('img/Varianza Explicada Acumulativa.png')
        st.write('Scatter PCA')
        pca_scatter = st.image('img/PCA.png')
        
elif opcion == "Modelo de Predicción":
    info_modelos = {
    "Logistic Regression": {
        "Mejores_parametros": "classifier__C: 0.1",
        "Mejor_precision": "0.6482103162001541"
    },
    "Gradient Boosting": {
        "Mejores_parametros": "classifier__n_estimators: 100 // classifier__max_depth: 5 // classifier__learning_rate: 0.1",
        "Mejor_precision": "0.6861271151839586"
    },
    "Random Forest": {
        "Mejores_parametros": "classifier__n_estimators: 200 // classifier__min_samples_split: 5 // classifier__max_depth: 20",
        "Mejor_precision": "0.6797287120627864"
    },
    "K-Neighbors Classifier": {
        "Mejores_parametros": "classifier__weights: 'distance' // classifier__n_neighbors: 3",
        "Mejor_precision": "0.6452397586535408"
    },
    "MLP Classifier": {
        "Mejores_parametros": "classifier__hidden_layer_sizes: (50,) // classifier__alpha: 0.001 // classifier__activation: 'relu'",
        "Mejor_precision": "0.6988867214081568"
        }
    }
    st.subheader("Aquí vamos a desplegar los modelos trabajados y haremos una predicción con nuestro mejor modelo.")
    st.header("Modelos supervisados")
    modelo_seleccionado = st.selectbox("Elige un modelo", list(info_modelos.keys()))
    modelo_info = info_modelos[modelo_seleccionado]
    st.write(modelo_info["Mejores_parametros"])
    st.write(modelo_info["Mejor_precision"])
    st.write('\n')
    st.subheader('El MLP Classifier fue el mejor modelo de SKlearn')
    st.write('Ahora veamos una confusion matrix para evaluar su desempeño')
    cm_sklearn = st.image('img/cm_sklearn.png')
    st.header('Modelo CNN')
    with st.expander('Modelo CNN'):
        st.write('Arquitectura del modelo')
        arquitectura = st.image('img/arquitectura_cnn.png')
        st.write('Evolución del modelo a través de los epochs')
        historico_modelo = st.image('img/history_model.png')
        st.write('Confusion Matrix')
        cm_cnn = st.image('img/cm_cnn.png')
        st.write('Errores en las predicciones')
        errores_predi = st.image('img/indices_incorrectos.png')
                    
elif opcion == 'Predicción en vivo':
    # Ruta al archivo del modelo guardado
    ruta_modelo = 'data/models/CNN_Best_Model_Rgb.h5'
    # Cargar el modelo
    modelo = load_model(ruta_modelo)
    archivo_imagen = st.file_uploader("Sube una imagen para la predicción", type=["jpg", "png", "jpeg"])
    if archivo_imagen is not None:
        # Convertir el archivo a un array de numpy
        file_bytes = np.asarray(bytearray(archivo_imagen.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Preprocesar la imagen (ajusta según sea necesario)
        img_resized = cv2.resize(img, (224, 224))  # Ajusta el tamaño según tu modelo
        img_array = np.array(img_resized) / 255.0  # Normalización
        img_array = img_array[np.newaxis, ...]  # Añadir una dimensión extra para el batch

        # Hacer la predicción
        predicciones = modelo.predict(img_array)
        # Mostrar la predicción
        st.write("Predicción:", predicciones)