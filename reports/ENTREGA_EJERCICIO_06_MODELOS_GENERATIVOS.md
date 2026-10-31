# INF-8239 · Ejercicio 06: Modelos Generativos (VAE y Autoencoder)

**Maestría en Inteligencia Artificial · Universidad Autónoma de Santo Domingo (UASD)**  
**Asignatura:** Ciencia de Datos II  
**Docente:** Edwin Ramón José Nolasco  
**Estudiante:** Juan Aníbal Pérez  
**Repositorio GitHub:** [https://github.com/sistemasperez/CienciaDatos2-U03.E06](https://github.com/sistemasperez/CienciaDatos2-U03.E06)  

---

## 1. Tabla Principal de Resultados Comparativos

La evaluación cuantitativa se realizó sobre las 10,000 imágenes del conjunto de prueba independiente de Fashion-MNIST ($28 \times 28 \times 1$), contrastando la compresión determinista del Autoencoder estándar frente al modelado probabilístico continuo del VAE:

| Modelo / Enfoque | Espacio Latente ($D$) | Parámetros | Tiempo Entrenamiento (s) | Reconstrucción Test (BCE) | Reconstrucción Test (MSE) | Regularización KL | Distancia Mínima a Train ($L_2$) | Diversidad Muestral | Tamaño Modelo (KB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Autoencoder Estándar (Baseline)** | 16 | 25,888 | 7.86 s | 0.3068 | 0.02029 | N/A (Discontinuo) | N/A (No Generativo) | N/A | 330.66 KB |
| **VAE (Variational Autoencoder)** | 2 | 202,516 | 12.44 s | 0.3405 | 0.03299 | 6.3018 | 1.6831 | 5.8485 | 415.66 KB |

---

## 2. Metodología, Datos y Reproducibilidad

### 2.1 Dataset y Preprocesamiento
El proyecto utiliza Fashion-MNIST (Zalando Research, licencia MIT), integrado por 70,000 imágenes en escala de grises de 10 categorías de indumentaria. La partición respeta el protocolo oficial: 60,000 imágenes para entrenamiento y 10,000 para evaluación fuera de muestra. 

Los tensores se normalizaron mediante división por 255.0, escalando los valores al intervalo continuo $[0.0, 1.0]$ con tipo de dato `float32`. La integridad de los contratos de entrada y salida se verificó mediante pruebas automatizadas en `tests/test_data.py`.

### 2.2 Formulación de las Arquitecturas
- **Autoencoder (Baseline):** Aplica una compresión determinista proyectando los 784 píxeles a un cuello de botella de 16 variables latentes mediante capas densas con activación ReLU y reconstrucción con función sigmoide, optimizada exclusivamente con entropía cruzada binaria (`binary_crossentropy`).
- **Variational Autoencoder (VAE):** Modela la distribución condicional $q_\phi(z|x)$ mapeando la entrada hacia dos vectores continuos en $\mathbb{R}^2$: el vector de medias $\mu$ y el logaritmo de varianzas $\log(\sigma^2)$. El paso hacia el decodificador utiliza el truco de reparametrización:
  $$z = \mu + \exp\left(\frac{1}{2}\log(\sigma^2)\right) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_2)$$
  La optimización minimiza la cota inferior variacional (ELBO), sumando el error de reconstrucción pixel a pixel y la divergencia de Kullback-Leibler frente al prior estándar $\mathcal{N}(0, I_2)$:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstrucción}} + \mathcal{L}_{\text{KL}}$$
  $$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{2} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

---

## 3. Análisis de Reconstrucción, Muestreo e Interpolación

Las diferencias entre ambos paradigmas quedaron documentadas en las figuras generadas en `reports/`:

1. **Reconstrucción vs Capacidad Generativa (`reports/ae_vae_panel.png`):**
   - El Autoencoder alcanza menores errores de reconstrucción (BCE = 0.3068, MSE = 0.02029) gracias a sus 16 dimensiones latentes y a la ausencia de restricciones probabilísticas. Sin embargo, su espacio latente está fragmentado: si se toma un punto arbitrario del espacio latente, el decodificador produce artefactos visuales sin sentido semántico.
   - El VAE cede nitidez (BCE = 0.3405, MSE = 0.03299) debido al cuello de botella extremo de 2 dimensiones y a la fuerza regularizadora de la divergencia KL. A cambio, adquiere capacidad generativa real: al muestrear vectores del prior $z \sim \mathcal{N}(0, I_2)$, el decodificador genera prendas sintéticas coherentes (calzados, abrigos y pantalones bien definidos).

2. **Interpolación Latente Continua (`reports/interpolation.png`):**
   - Se evaluó una trayectoria lineal continua entre dos puntos extremos del espacio latente: $z_A = (-2.0, -1.0)$ y $z_B = (2.0, 1.0)$.
   - La transición generada muestra una metamorfosis suave entre clases de prendas, confirmando que la regularización KL previene zonas vacías y garantiza densidad de probabilidad en todo el plano latente.

3. **Mapeo del Espacio Latente 2D (`reports/latent_space_2d.png`):**
   - La proyección de las 10 clases de prueba sobre el plano $(z_1, z_2)$ revela que el VAE agrupa espontáneamente ítems semánticamente afines. Los tipos de calzado (sandalias, zapatillas y botas) se concentran en una región común, claramente separada de pantalones y vestidos.

---

## 4. Fidelidad, Diversidad y Evaluación de No-Memorización

Para comprobar que el VAE no memoriza ni reproduce copias idénticas del conjunto de entrenamiento, se diseñó una auditoría cuantitativa basada en vecinos más cercanos (Nearest Neighbors) sobre 500 muestras sintéticas generadas:

- **Distancia euclidiana media al vecino más cercano de entrenamiento:** 2.9571.
- **Distancia euclidiana mínima observada:** 1.6831.
- **Diversidad intra-muestral (distancia promedio entre pares generados):** 5.8485.

El hecho de que la distancia euclidiana mínima sea estrictamente positiva y comparable a la separación natural entre imágenes reales confirma que el modelo no opera como una tabla de búsqueda ni sobreajusta instancias de entrenamiento. Produce patrones novedosos consistentes con la distribución marginal aprendida.

---

## 5. Costo Computacional, Límites y Conclusiones

### 5.1 Costo de Entrenamiento y Almacenamiento
- El Autoencoder requirió 7.86 segundos para completar 6 épocas en CPU, generando un archivo serializado de 330.66 KB (25,888 parámetros).
- El VAE tomó 12.44 segundos para 8 épocas, con un peso de 415.66 KB para el decodificador generativo (202,516 parámetros totales entre encoder y decoder).
- La eficiencia del modelo permite inferencia inmediata en CPU y su integración en la interfaz de usuario desarrollada en Streamlit (`app/streamlit_app.py`).

### 5.2 Límites Técnicos Identificados
1. **Borrosidad por compresión y función de pérdida:** El uso de 2 dimensiones latentes facilita la visualización e interactividad directa, pero comprime drásticamente la variabilidad de alta frecuencia. La combinación de entropía cruzada pixel a pixel tiende a promediar configuraciones posibles, produciendo bordes difuminados.
2. **Supuesto de independencia en el prior:** Asumir una distribución a priori gaussiana isotrópica $\mathcal{N}(0, I_2)$ penaliza correlaciones complejas en la geometría de los datos reales, generando deformaciones ligeras en los bordes del dominio latente ($|z| > 2.5$).
3. **Advertencia ética de privacidad:** La generación sintética no garantiza anonimato por sí misma. Sin mecanismos formales como Privacidad Diferencial (DP-SGD), los modelos generativos pueden retener información implícita de los datos de entrenamiento.

---

## 6. Declaración de Uso Ético y Responsable de IA

En concordancia con los requerimientos académicos del curso, se declaran los siguientes aspectos:

- **Herramienta asistencial:** Gemini 3.8 Flash (High) mediante Antigravity IDE.
- **Alcance del apoyo:**
  1. Diseño y verificación de las pruebas unitarias automatizadas (`tests/test_models.py` y `tests/test_artifacts.py`).
  2. Implementación vectorizada de la métrica de vecinos más cercanos para la auditoría de memorización.
  3. Automatización de la exportación del notebook reproducible (`notebooks/lab10_generativos.ipynb`).
- **Verificaciones técnicas realizadas:**
  - Se confirmó analíticamente la formulación del gradiente estocástico y el término KL en `train_step`.
  - Se comprobó la normalización estricta de imágenes en $[0, 1]$ y la consistencia dimensional de tensores en inferencia.
  - Se ejecutó la suite completa de pruebas unitarias (`uv run pytest`), aprobando 8 de 8 casos de prueba.
- **Responsabilidad asumida:** El estudiante responde por el código, la ejecución de los experimentos, la interpretación de resultados y la defensa de las conclusiones presentadas.
