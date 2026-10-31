# Model Card · Autoencoder y VAE

## 1. Modelos y Versiones
- **Nombre de los Modelos:** Autoencoder Determinista (Baseline) y Variational Autoencoder (VAE).
- **Versión:** 1.0.0 (INF-8239 · Unidad 03 · LAB10 · Ejercicio 06).
- **Fecha de Evaluación:** Octubre 2026.
- **Tipo de Modelos:** Redes neuronales para compresión y aprendizaje de representaciones generativas no supervisadas.
- **Framework y Entorno:** TensorFlow 2.21.0 / Keras 3.15.1, Python 3.12, empaquetado y gestionado con `uv`.
- **Repositorio de Referencia:** [https://github.com/sistemasperez/CienciaDatos2-U03.E06](https://github.com/sistemasperez/CienciaDatos2-U03.E06).

---

## 2. Uso Previsto y Usuarios
- **Propósito Principal:** Contrastar empíricamente la compresión latente determinista frente a la inferencia variacional probabilística continua sobre imágenes monocromáticas de prendas de vestir.
- **Casos de Uso Válidos:**
  - Reducción de dimensionalidad y visualización interactiva del espacio latente en dos dimensiones ($z_1, z_2$).
  - Interpolación lineal continua entre categorías semánticas de indumentaria.
  - Generación de muestras sintéticas exploratorias a partir de un prior gaussiano $\mathcal{N}(0, I_2)$.
- **Público Objetivo:** Estudiantes, investigadores y profesionales en Ciencia de Datos e Inteligencia Artificial con fines docentes y de experimentación.

---

## 3. Usos Fuera de Alcance y Riesgos Éticos
- **No apto para diseño fotorrealista:** Las arquitecturas son deliberadamente compactas para fines didácticos; no deben emplearse en producción comercial de catálogos o diseño asistido de alta resolución.
- **Riesgo Ético de Fuga y Privacidad:** La síntesis generativa **no garantiza anonimato por el solo hecho de ser generada sintéticamente**. Si se entrenan modelos generativos con datos sensibles sin Privacidad Diferencial (DP-SGD), los modelos pueden memorizar y filtrar atributos de los datos originales.
- **Inferencia en Hardware Restringido:** Diseñado para CPU estándar sin requerir aceleradores GPU dedicados.

---

## 4. Descripción del Dataset, Campos, Licencia y Riesgo de Fuga de Datos

### 4.1 Procedencia, Origen y Licencia
- **Nombre:** Fashion-MNIST.
- **Creador / Procedencia:** Zalando Research (Han Xiao, Kashif Rasul, Roland Vollgraf, 2017).
- **Motivación Original:** Desarrollado como un sustituto directo, moderno y más exigente del MNIST clásico de dígitos manuscritos, manteniendo idéntico formato de tensores y particiones.
- **Licencia:** MIT License (permite uso académico y comercial libre con la correspondiente atribución).

### 4.2 Esquema de Datos y Descripción de Campos
El dataset se compone de imágenes individuales de prendas de vestir en escala de grises centradas sobre fondo negro:

| Campo / Variable | Tipo de Dato Original | Tipo Transformado | Rango de Valores | Dimensiones / Forma | Descripción Detallada |
|---|:---:|:---:|:---:|:---:|---|
| **`images` (Tensores de Entrada)** | `uint8` | `float32` | $[0.0, 1.0]$ | $(N, 28, 28, 1)$ | Matriz monocanal de $28 \times 28$ píxeles. El valor 0.0 representa fondo negro y 1.0 blanco puro. |
| **`labels` (Categoría de la Prenda)** | `uint8` | `int64` | $\{0, 1, \dots, 9\}$ | $(N,)$ | Identificador entero de la clase a la que pertenece la prenda representada en la imagen. |

#### Diccionario de Clases (10 categorías balanceadas):
- **Clase 0 — `T-shirt/top`:** Camisetas, tops y remeras básicas de manga corta o sin mangas.
- **Clase 1 — `Trouser`:** Pantalones de vestir, jeans y pantalones deportivos largos.
- **Clase 2 — `Pullover`:** Suéteres, jerseys y prendas de punto de manga larga.
- **Clase 3 — `Dress`:** Vestidos de una sola pieza de silueta femenina.
- **Clase 4 — `Coat`:** Abrigos, gabardinas y sacos pesados de abrigo exterior.
- **Clase 5 — `Sandal`:** Sandalias abiertas y calzado ligero de verano.
- **Clase 6 — `Shirt`:** Camisas formales e informales con cuello abotonado.
- **Clase 7 — `Sneaker`:** Zapatillas deportivas y calzado atlético cerrado.
- **Clase 8 — `Bag`:** Bolsos de mano, carteras, mochilas y maletines.
- **Clase 9 — `Ankle boot`:** Botines cerrados y botas de caña corta.

### 4.3 Partición de Datos y Distribución
- **Conjunto de Entrenamiento (`train`):** 60,000 imágenes ($28 \times 28 \times 1$), exactamente 6,000 imágenes por clase (distribución perfectamente balanceada).
- **Conjunto de Prueba (`test`):** 10,000 imágenes ($28 \times 28 \times 1$), exactamente 1,000 imágenes por clase.
- Las particiones están fijadas formalmente desde el repositorio oficial de Zalando Research, impidiendo variabilidad en los splits entre diferentes laboratorios.

### 4.4 Riesgo y Prevención de Fuga de Datos (*Data Leakage*)
1. **Fuga en Preprocesamiento:**
   - La normalización se realiza mediante una transformación determinista escalar fija ($x / 255.0$). Al no calcular estadísticas empíricas (como media o desviación estándar globales) dependientes del conjunto de prueba, se garantiza la total independencia estadística entre conjuntos.
2. **Aislamiento de la Optimización:**
   - Durante las épocas de entrenamiento de ambos modelos (Autoencoder y VAE), únicamente se utilizaron los tensores de `x_train`.
   - En el VAE, la regularización KL empuja la distribución posterior $q(z|x)$ hacia el prior analítico $\mathcal{N}(0, I_2)$, sin usar información alguna de `x_test`.
   - El conjunto de prueba `x_test` se reservó de forma estricta para el cálculo final de las métricas de reconstrucción (BCE y MSE fuera de muestra).
3. **Auditoría de Fuga por Memorización:**
   - Se ejecutó un análisis de vecinos más cercanos (Nearest Neighbors) entre las muestras generadas y el conjunto de entrenamiento para verificar que el VAE no hubiera memorizado imágenes de train, confirmando una distancia euclidiana mínima de **1.6831** (sin duplicados exactos).

---

## 5. Arquitectura de los Modelos y Espacio Latente

```text
[Autoencoder Baseline]
Entrada (28, 28, 1) ---> Flatten(784) ---> Dense(16, ReLU) ---> Dense(784, Sigmoid) ---> Reshape(28, 28, 1)
                                                 ^
                                          Espacio Latente (D=16)

[Variational Autoencoder (VAE)]
                     ┌---> Dense(2) [Vector de Medias μ] --------┐
Entrada (28, 28, 1) -┼---> Flatten(784) -> Dense(128, ReLU)     ├-> Reparametrización: z = μ + exp(0.5*logσ²)*ε
                     └---> Dense(2) [Log-Varianza log(σ²)] ------┘         |
                                                                           v
                                                          Espacio Latente Continuo (D=2)
                                                                           |
Decodificador: Latente (2) ---> Dense(128, ReLU) ---> Dense(784, Sigmoid) ---> Salida (28, 28, 1)
```

- **Parámetros del Autoencoder:** 25,888 parámetros entrenables.
- **Parámetros del VAE:** Encoder: 100,996; Decoder: 101,520; Total: 202,516 parámetros.

---

## 6. Métricas de Reconstrucción y Desempeño Cuantitativo

Resultados evaluados sobre las 10,000 imágenes del conjunto de prueba:

| Métrica de Evaluación | Autoencoder ($D=16$) | VAE ($D=2$) | Justificación Técnica |
|---|:---:|:---:|---|
| **Binary Cross-Entropy (BCE)** | 0.3068 | 0.3405 | El AE retiene mayor agudeza por disponer de 8 veces más dimensiones latentes y no tener restricción KL. |
| **Mean Squared Error (MSE)** | 0.02029 | 0.03299 | El VAE promedia configuraciones posibles en 2D, produciendo una reconstrucción más difusa pero continua. |
| **Divergencia KL Media** | N/A | 6.3018 | Mide el coste de aproximar la distribución posterior al prior estándar $\mathcal{N}(0, I_2)$. |

---

## 7. Diversidad y Evaluación de Memorización (Nearest Neighbors)

Para verificar si las imágenes generadas por el decodificador a partir de $z \sim \mathcal{N}(0, I_2)$ son originales o producto de sobreajuste:
- **Distancia euclidiana media a la muestra de entrenamiento más cercana:** 2.9571.
- **Distancia euclidiana mínima observada:** 1.6831.
- **Diversidad intra-muestral (distancia promedio entre pares generados):** 5.8485.
- **Dictamen:** No existe duplicación memorística; el modelo aprendió una variedad suave que interpola atributos reales.

---

## 8. Costos Computacionales y Especificaciones de Hardware
- **Entorno de Ejecución:** CPU x86_64, Windows, Python 3.12 (`uv`).
- **Tiempos de Entrenamiento:**
  - Autoencoder (6 épocas, batch size 256): 7.86 segundos.
  - VAE (8 épocas, batch size 256): 12.44 segundos.
- **Huella de Almacenamiento:**
  - `autoencoder.keras`: 330.66 KB.
  - `decoder.keras`: 415.66 KB.

---

## 9. Limitaciones Técnicas y Recomendaciones
1. **Borrosidad Estructural:** El VAE con $D=2$ y pérdida por entropía cruzada tiende a suavizar bordes de prendas complejas (como estampados o cierres metálicos).
2. **Prior Estándar:** La hipótesis de independencia entre $z_1$ y $z_2$ genera una ligera pérdida de definición en las regiones extremas del espacio latente ($|z| > 2.5$).
3. **Privacidad:** Reiterar que la generación sintética requiere protocolos de privacidad diferencial si los datos de origen contienen información sensible.
