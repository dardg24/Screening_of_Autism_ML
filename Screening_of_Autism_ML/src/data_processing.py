#Librerias

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os



def create_image_dataframe():
    """
    Esta función crea un DataFrame que contiene las rutas de las imágenes y sus respectivas etiquetas.
    Las imágenes se recogen de las carpetas 'autistic' y 'non_autistic', ubicadas en el directorio actual.
    
    Returns:
        df (pd.DataFrame): DataFrame con dos columnas - 'image_path' y 'label'.
    """
    # Rutas relativas a las carpetas de imágenes
    autistic_dir = 'autistic'
    non_autistic_dir = 'non_autistic'

    image_paths = []
    labels = []

    # Carpeta autistic
    for image_name in os.listdir(autistic_dir):
        image_paths.append(os.path.join(autistic_dir, image_name))
        labels.append('autistic')

    # Carpeta non_autistic
    for image_name in os.listdir(non_autistic_dir):
        image_paths.append(os.path.join(non_autistic_dir, image_name))
        labels.append('non_autistic')

    # Crear el DataFrame
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})

    return df

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



df = create_image_dataframe()

prepare_dataframe(df)