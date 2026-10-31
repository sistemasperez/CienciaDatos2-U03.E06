# Model Card · Autoencoder y VAE

## 1. Modelos y Versiones
- **Nombre:** Autoencoder Determinista (Baseline) y Variational Autoencoder (VAE).
- **Versión:** 1.0.0 (INF-8239 · Unidad 03 · LAB10).
- **Fecha de evaluación:** Octubre 2026.
- **Tipo de modelo:** Redes neuronales para compresión y aprendizaje de representaciones no supervisadas / generativas.
- **Framework:** TensorFlow 2.21.0 / Keras 3.15.1 en Python 3.12 (`uv`).

## 2. Uso Previsto y Usuarios
- **Propósito:** Análisis comparativo entre compresión latente determinista y modelado generativo continuo sobre imágenes monocromáticas de baja resolución ($28 \times 28$).
- **Casos de uso válidos:**
  - Reducción de dimensionalidad y visualización didáctica del espacio latente en dos dimensiones ($z_1, z_2$).
  - Interpolación lineal continua entre categorías visuales de prendas de vestir.
  - Generación de muestras sintéticas exploratorias a partir de un prior gaussiano $\mathcal{N}(0, I_2)$.
- **Usuarios destinatarios:** Investigadores, docentes y estudiantes de Ciencia de Datos e Inteligencia Artificial.

## 3. Usos Fuera de Alcance
- No apto para generación fotorrealista de alta fidelidad ni aplicaciones comerciales de diseño de indumentaria.
- **Riesgo ético de privacidad:** No debe asumirse que las muestras sintéticas son anónimas o privadas por el solo hecho de ser generadas; los modelos generativos profundos pueden filtrar patrones si no se implementa Privacidad Diferencial (DP-SGD).
- No diseñado para inferencia en tiempo real en dispositivos embebidos ultra-restringidos sin cuantización previa.

## 4. Dataset, Licencia y Particiones
- **Dataset:** Fashion-MNIST (Zalando Research).
- **Licencia:** MIT License.
- **Estructura y Cardinalidad:**
  - Entrenamiento: 60,000 imágenes en escala de grises ($28 \times 28 \times 1$).
  - Prueba: 10,000 imágenes independientes para evaluación fuera de muestra.
  - Clases (10 categorías balanceadas): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- **Preprocesamiento:** Normalización estricta por división entre 255.0, convirtiendo los píxeles a valores reales en el rango continuo $[0.0, 1.0]$ con tipado `float32`.

## 5. Arquitectura y Espacio Latente
- **Autoencoder Estándar:**
  - Entrada: $(28, 28, 1) \rightarrow \text{Flatten}(784)$.
  - Capa latente: $\text{Dense}(16, \text{activation}=\text{"relu"})$ con compresión a $D=16$.
  - Decodificador: $\text{Dense}(784, \text{activation}=\text{"sigmoid"}) \rightarrow \text{Reshape}(28, 28, 1)$.
  - Total de parámetros: 25,888.
- **Variational Autoencoder (VAE):**
  - **Encoder:** $\text{Input}(28, 28, 1) \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow$ Dos ramas paralelas de proyección lineal:
    - $\mu$: $\text{Dense}(2)$ para el vector de medias.
    - $\log(\sigma^2)$: $\text{Dense}(2)$ para el logaritmo de las varianzas.
    - Parámetros del encoder: 100,996.
  - **Reparametrización:** $z = \mu + \exp(0.5 \cdot \log(\sigma^2)) \odot \epsilon$, con $\epsilon \sim \mathcal{N}(0, I_2)$.
  - **Decoder:** $\text{Input}(2) \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{Dense}(784, \text{Sigmoid}) \rightarrow \text{Reshape}(28, 28, 1)$.
    - Parámetros del decoder: 101,520.
  - Total de parámetros VAE: 202,516.

## 6. Métricas de Reconstrucción y Generación (Test Set)
Evaluadas sobre las 10,000 imágenes de prueba de Fashion-MNIST:

| Métrica | Autoencoder ($D=16$) | VAE ($D=2$) | Interpretación |
|---|:---:|:---:|---|
| **Binary Cross-Entropy (BCE)** | 0.3068 | 0.3405 | AE reconstruye con mayor fidelidad por mayor dimensionalidad latente y ausencia de penalización KL. |
| **Mean Squared Error (MSE)** | 0.02029 | 0.03299 | VAE exhibe reconstrucciones más suaves/borrosas debido al cuello de botella 2D. |
| **Divergencia KL Media** | N/A | 6.3018 | Mide la regularización hacia el prior $\mathcal{N}(0, I_2)$, evitando agujeros en el espacio latente. |

## 7. Diversidad y Evaluación de Vecinos Cercanos (No-Memorización)
Para evaluar si el VAE generó muestras originales o simplemente memorizó datos de entrenamiento, se muestrearon 500 vectores aleatorios $z \sim \mathcal{N}(0, I_2)$ y se calculó la distancia euclidiana mínima frente a 5,000 ejemplos de entrenamiento:
- **Distancia euclidiana media al vecino más cercano:** 2.9571.
- **Distancia euclidiana mínima al vecino más cercano:** 1.6831.
- **Diversidad intra-muestral (Pairwise $L_2$):** 5.8485.
- **Conclusión de memoria:** Al registrar una distancia mínima estrictamente positiva y relativamente alta, se descarta memorización puntual o copia directa de instancias de entrenamiento; el modelo sintetiza combinaciones continuas válidas.

## 8. Hardware, Tiempo y Tamaño en Disco
- **Hardware:** CPU x86_64 multinúcleo en entorno Windows.
- **Tiempo de entrenamiento:**
  - Autoencoder (6 épocas, batch size 256): 7.86 segundos.
  - VAE (8 épocas, batch size 256): 12.44 segundos.
- **Tamaño de artefactos exportados:**
  - `autoencoder.keras`: 330.66 KB.
  - `decoder.keras`: 415.66 KB.

## 9. Limitaciones, Riesgos y Supervisión
1. **Borrosidad de Reconstrucción:** La combinación de una función de pérdida basada en entropía cruzada pixel a pixel junto con un espacio latente ultracomprimido de 2 dimensiones suprime texturas finas (ej. estampados de camisetas o cordones de calzados).
2. **Prior Estándar:** La suposición de componentes no correlacionadas en $\mathcal{N}(0, I_2)$ puede generar muestras atípicas en zonas periféricas del espacio latente ($|z| > 3$).
3. **Supervisión Ética:** Toda utilización de datos sintéticos derivados debe documentar su procedencia algorítmica y no pretender sustituir auditorías formales de privacidad.
