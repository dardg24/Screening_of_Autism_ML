import sys
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers


IMAGE_WIDTH= 224
IMAGE_HEIGHT= 224
IMAGE_CHANNELS= 3
IMAGE_SIZE= (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
BATCH_SIZE = 32
EPOCHS = 20

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


train = pd.read_csv('Screening_of_Autism_ML_project_final/data/train/train.csv')
train_p = prepare_dataframe(train)

print (train_p)


X_train = np.stack(train_p['Imagenes'].values)
y_train = np.array(train_p['target'])



print (X_train.shape)
print (y_train.shape)

earlystop = EarlyStopping(patience=5)
mcheckpoint = ModelCheckpoint("callback_model_3_rgb.h5")

optimizer = Adam(learning_rate=0.0005)  
layers = [
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=IMAGE_SIZE),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(1, activation='sigmoid')
]

model_3 = keras.Sequential(layers)
model_3.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])

history_3 = model_3.fit(X_train,
                        y_train,
                        epochs = EPOCHS,
                        batch_size = BATCH_SIZE,
                        callbacks = [earlystop, mcheckpoint],
                        validation_split = 0.2
    
)