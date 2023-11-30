import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def prepare_dataframe(df):
    """
    Esta función procesa un DataFrame agregando una columna 'target' mapeada de la columna 'label',
    procesa las imágenes de las rutas dadas en 'image_path' y reorganiza el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original con columnas 'image_path' y 'label'.

    Returns:
        pd.DataFrame: DataFrame procesado con columnas adicionales 'target' e 'Imagenes'.
    """
    # Mapeo de etiquetas a valores numéricos
    target_map = {'autistic': 0, 'non_autistic': 1}
    df['target'] = df['label'].map(target_map)

    # Función para leer y escalar imágenes
    def procesar_imagen(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_scale = np.array(img) / 255.0
        return img_scale

    # Aplicar la función de procesamiento de imágenes
    df['Imagenes'] = df['image_path'].apply(procesar_imagen)

    # Mezclar y reiniciar índice del DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    return df


# Declaro los DF
test = pd.read_csv('Screening_of_Autism_ML_project_final/data/test/test.csv')
test_p = prepare_dataframe(test)
print (test_p)

# Declaro los test
X_test = np.stack(test_p['Imagenes'].values)
y_test = np.array(test_p['target'])
print (X_test.shape)
print (y_test.shape)

# Cargar el modelo
ruta_modelo = 'Screening_of_Autism_ML_project_final\models\CNN_Best_Model_Rgb.h5'
modelo = load_model(ruta_modelo)

# Prediccion del modelo
evaluacion = modelo.evaluate(X_test, y_test)
predicciones_probabilidades = modelo.predict(X_test)
predicciones = (predicciones_probabilidades > 0.5).astype("int32").squeeze()
print (evaluacion)
print (predicciones)

# Matriz de confusión para el conjunto  (normalizada)
# Configurar subplots
fig, ax = plt.subplots(2, 2, figsize=(15, 12))
sns.heatmap(confusion_matrix(y_test, predicciones, normalize='true'), annot=True, fmt='.2%', ax=ax[0, 1], cmap='Blues')
ax[0, 1].set_title('Matriz de Confusión (Entrenamiento, Normalizada)')
ax[0, 1].set_ylabel('Etiqueta Real')
ax[0, 1].set_xlabel('Etiqueta Predicha')

# Matriz de confusión para el conjunto  (sin normalizar)
sns.heatmap(confusion_matrix(y_test, predicciones), annot=True, fmt='d', ax=ax[1, 0])
ax[1, 0].set_title('Matriz de Confusión (Prueba, Sin Normalizar)')
ax[1, 0].set_ylabel('Etiqueta Real')
ax[1, 0].set_xlabel('Etiqueta Predicha')