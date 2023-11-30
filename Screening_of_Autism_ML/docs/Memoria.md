
# Memoria del Proyecto: Screening of Autism

## Introducción
El presente proyecto se centra en el desarrollo de modelos de machine learning para identificar rasgos del espectro autista en niños a través del análisis de imágenes. Esta iniciativa busca aportar en la detección temprana y el diagnóstico del autismo, una condición que impacta a un número significativo de individuos a nivel mundial.

## Análisis Exploratorio de Datos (EDA)
El EDA realizado tuvo como objetivo comprender mejor el conjunto de datos con el que se trabajaría. Este proceso incluyó:

- **Análisis de la distribución de las clases**: Se observó cómo estaban distribuidas las categorías (autista y no autista) dentro del conjunto de datos.
- **Exploración de las características de las imágenes**: Se analizó el tamaño, la escala de colores y otras propiedades relevantes de las imágenes.
- **Identificación de posibles desafíos**: Se reconocieron los desafíos inherentes al conjunto de datos, como el desbalance de clases o la calidad de las imágenes.

## Modelos de Sklearn - PCA y Modelos Supervisados
Inicialmente, se implementaron modelos supervisados para establecer una línea base (baseline). Esto incluyó:

- **PCA (Análisis de Componentes Principales)**: Se utilizó PCA para la reducción de dimensionalidad, lo que ayudó a simplificar los modelos sin perder características esenciales.
- **Cinco Modelos Supervisados**: Se entrenaron varios modelos, como Regresión Logística, Random Forest, SVM, K-Nearest Neighbors y Decision Tree, para comparar su rendimiento y seleccionar el más adecuado.

## Desarrollo y Experimentación con Redes Neuronales Convolucionales (CNN)
La fase siguiente se centró en el desarrollo y experimentación con CNN, donde:

- **Diseño de la Arquitectura CNN**: Se diseñaron y probaron diversas arquitecturas de CNN, ajustando parámetros como el número de capas, filtros y neuronas.
- **Experimentación y Ajuste Fino**: Se realizaron más de 10 experimentos con diferentes configuraciones de CNN para maximizar la precisión y minimizar el sobreajuste.
- **Evaluación de Modelos**: Cada modelo fue rigurosamente evaluado, utilizando métricas como la precisión (accuracy) y la pérdida (loss).

## Resultados
El modelo ganador de CNN demostró ser efectivo, alcanzando una precisión del 78-80% en el conjunto de test. Estos resultados son prometedores y sugieren que el modelo podría ser una herramienta útil en la detección temprana del autismo.

## Conclusiones y Futuros Trabajos
Este proyecto ha demostrado el potencial del machine learning y las CNN en el campo de la detección del autismo. Para trabajos futuros, se contempla:

- **Ampliación del Conjunto de Datos**: Incorporar un conjunto de datos más grande y diverso para mejorar la robustez del modelo.
- **Experimentación con Otras Arquitecturas de CNN**: Probar con arquitecturas más avanzadas o recientes.
- **Desarrollo de una Aplicación Web**: Crear una aplicación interactiva para que los profesionales puedan utilizar el modelo en un entorno práctico.

## Autor
- Daniel Rendon Gouveia
