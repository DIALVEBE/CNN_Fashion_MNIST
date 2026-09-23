import os

# Usar CPU y reducir los mensajes internos de TensorFlow.
# Estas variables deben declararse antes de importar TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# 1. Configuración
# --------------------------------------------------

SEMILLA = 42
EPOCAS = 10
TAMANO_LOTE = 128

np.random.seed(SEMILLA)
tf.keras.utils.set_random_seed(SEMILLA)

NOMBRES_CLASES = [
    "Camiseta",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]

DIRECTORIO_RESULTADOS = Path("resultados")
DIRECTORIO_MODELOS = Path("modelos")

DIRECTORIO_RESULTADOS.mkdir(exist_ok=True)
DIRECTORIO_MODELOS.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Descargar el dataset desde Kaggle
# --------------------------------------------------

print("Descargando Fashion MNIST desde Kaggle...")

ruta_dataset = Path(
    kagglehub.dataset_download(
        "zalando-research/fashionmnist"
    )
)

print(f"Dataset almacenado en: {ruta_dataset}")

archivo_entrenamiento = (
    ruta_dataset / "fashion-mnist_train.csv"
)

archivo_prueba = (
    ruta_dataset / "fashion-mnist_test.csv"
)

if not archivo_entrenamiento.exists():
    raise FileNotFoundError(
        f"No se encontró: {archivo_entrenamiento}"
    )

if not archivo_prueba.exists():
    raise FileNotFoundError(
        f"No se encontró: {archivo_prueba}"
    )


# --------------------------------------------------
# 3. Cargar los datos
# --------------------------------------------------

print("\nCargando los archivos CSV...")

datos_entrenamiento = pd.read_csv(archivo_entrenamiento)
datos_prueba = pd.read_csv(archivo_prueba)

print(
    "Datos de entrenamiento:",
    datos_entrenamiento.shape
)

print(
    "Datos de prueba:",
    datos_prueba.shape
)

print("\nPrimeras columnas:")
print(datos_entrenamiento.iloc[:5, :8])


# --------------------------------------------------
# 4. Separar imágenes y etiquetas
# --------------------------------------------------

y_train = datos_entrenamiento["label"].to_numpy()
x_train = datos_entrenamiento.drop(
    columns=["label"]
).to_numpy()

y_test = datos_prueba["label"].to_numpy()
x_test = datos_prueba.drop(
    columns=["label"]
).to_numpy()

print("\nForma original de entrenamiento:")
print(x_train.shape)


# --------------------------------------------------
# 5. Preparar las imágenes
# --------------------------------------------------

# Convertir cada fila de 784 números en una imagen
# de 28 x 28 píxeles con un canal de color.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Los píxeles están entre 0 y 255.
# Los normalizamos para dejarlos entre 0 y 1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nForma después de reconstruir las imágenes:")
print(x_train.shape)

print("\nValor mínimo:", x_train.min())
print("Valor máximo:", x_train.max())


# --------------------------------------------------
# 6. Mostrar algunas imágenes
# --------------------------------------------------

figura, ejes = plt.subplots(2, 5, figsize=(12, 5))

for indice, eje in enumerate(ejes.flat):
    eje.imshow(
        x_train[indice].squeeze(),
        cmap="gray"
    )

    etiqueta = y_train[indice]

    eje.set_title(
        NOMBRES_CLASES[etiqueta]
    )

    eje.axis("off")

plt.tight_layout()

plt.savefig(
    DIRECTORIO_RESULTADOS / "muestras.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# 7. Crear la red neuronal convolucional
# --------------------------------------------------

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        units=128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        units=10,
        activation="softmax"
    ),
])

modelo.summary()


# --------------------------------------------------
# 8. Configurar el entrenamiento
# --------------------------------------------------

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

detencion_temprana = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

guardar_mejor_modelo = tf.keras.callbacks.ModelCheckpoint(
    filepath=DIRECTORIO_MODELOS / "fashion_mnist.keras",
    monitor="val_loss",
    save_best_only=True,
)


# --------------------------------------------------
# 9. Entrenar
# --------------------------------------------------

print("\nIniciando entrenamiento...")

historial = modelo.fit(
    x_train,
    y_train,
    epochs=EPOCAS,
    batch_size=TAMANO_LOTE,
    validation_split=0.2,
    callbacks=[
        detencion_temprana,
        guardar_mejor_modelo,
    ],
)


# --------------------------------------------------
# 10. Evaluar con datos no utilizados
# --------------------------------------------------

perdida, exactitud = modelo.evaluate(
    x_test,
    y_test,
    verbose=0,
)

print(f"\nPérdida en prueba: {perdida:.4f}")
print(f"Exactitud en prueba: {exactitud:.4f}")


# --------------------------------------------------
# 11. Realizar predicciones
# --------------------------------------------------

probabilidades = modelo.predict(
    x_test,
    verbose=0
)

predicciones = np.argmax(
    probabilidades,
    axis=1
)

print("\nReporte de clasificación:")

print(
    classification_report(
        y_test,
        predicciones,
        target_names=NOMBRES_CLASES,
    )
)


# --------------------------------------------------
# 12. Matriz de confusión
# --------------------------------------------------

matriz = confusion_matrix(
    y_test,
    predicciones
)

plt.figure(figsize=(11, 8))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=NOMBRES_CLASES,
    yticklabels=NOMBRES_CLASES,
)

plt.xlabel("Predicción de la red")
plt.ylabel("Respuesta correcta")
plt.title("Matriz de confusión")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    DIRECTORIO_RESULTADOS / "matriz_confusion.png",
    dpi=150,
)

plt.show()


# --------------------------------------------------
# 13. Gráfica del entrenamiento
# --------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.plot(
    historial.history["accuracy"],
    label="Entrenamiento",
)

plt.plot(
    historial.history["val_accuracy"],
    label="Validación",
)

plt.xlabel("Época")
plt.ylabel("Exactitud")
plt.title("Exactitud del modelo")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    historial.history["loss"],
    label="Entrenamiento",
)

plt.plot(
    historial.history["val_loss"],
    label="Validación",
)

plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Pérdida del modelo")
plt.legend()

plt.tight_layout()

plt.savefig(
    DIRECTORIO_RESULTADOS / "entrenamiento.png",
    dpi=150,
)

plt.show()


# --------------------------------------------------
# 14. Mostrar algunas predicciones
# --------------------------------------------------

figura, ejes = plt.subplots(2, 5, figsize=(12, 6))

for indice, eje in enumerate(ejes.flat):
    prediccion = predicciones[indice]
    respuesta_real = y_test[indice]

    color = (
        "green"
        if prediccion == respuesta_real
        else "red"
    )

    eje.imshow(
        x_test[indice].squeeze(),
        cmap="gray",
    )

    eje.set_title(
        f"Predicción: {NOMBRES_CLASES[prediccion]}\n"
        f"Real: {NOMBRES_CLASES[respuesta_real]}",
        color=color,
    )

    eje.axis("off")

plt.tight_layout()

plt.savefig(
    DIRECTORIO_RESULTADOS / "predicciones.png",
    dpi=150,
)

plt.show()
