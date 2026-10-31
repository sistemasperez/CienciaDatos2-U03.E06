# Declaración de uso de inteligencia artificial · INF-8239 (Ejercicio 06)

## 1. Herramientas utilizadas
- Antigravity IDE y modelo asistencial Gemini 3.8 Flash (High).
- Python 3.12, `uv`, TensorFlow 2.21, Keras 3.15, scikit-learn y pytest.

## 2. Propósito de cada uso
- **Estructuración de pruebas automatizadas:** Asistencia en la formulación de casos de prueba unitarios en `pytest` para contratos de entrada/salida de tensores y continuidad en la interpolación lineal.
- **Implementación de métricas de evaluación:** Apoyo en el diseño del cálculo de distancias al vecino más cercano (Nearest Neighbors) para auditar empíricamente la no-memorización de imágenes de entrenamiento.
- **Generación del notebook reproducible:** Asistencia técnica en la automatización del archivo `.ipynb` con celdas ejecutadas y visualizaciones embebidas.

## 3. Prompts o decisiones relevantes
- Selección de la dimensionalidad latente del VAE en $D=2$ para posibilitar la visualización directa en scatter plot y la exploración intuitiva en la aplicación Streamlit sin necesidad de proyecciones adicionales (PCA/t-SNE).
- Inclusión explícita de la advertencia metodológica en el Model Card: la síntesis generativa no equivale automáticamente a privacidad de datos.
- Determinismo en TensorFlow mediante `TF_DETERMINISTIC_OPS=1` y fijación de semilla aleatoria (`SEED = 42`) para garantizar reproducibilidad exacta.

## 4. Elementos verificados por el equipo
- Comprobación matemática del término de divergencia de Kullback-Leibler:
  $$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum \left(1 + \log(\sigma^2) - \mu^2 - \sigma^2\right)$$
- Verificación del correcto funcionamiento del truco de reparametrización durante el paso de entrenamiento (`train_step`).
- Validación de que los modelos guardados (`.keras`) se carguen y ejecuten inferencia sin errores de serialización.
- Ejecución completa de la suite de pruebas unitarias (`uv run pytest`), con 8 pruebas aprobadas al 100%.

## 5. Cambios realizados sobre resultados generados
- Ajuste del cálculo de diversidad inter-muestra para evitar comparar pares idénticos consigo mismos.
- Incorporación de etiquetas de clase de Fashion-MNIST en el gráfico de dispersión latente 2D para enriquecer la interpretabilidad semántica.

## 6. Responsabilidad asumida
El estudiante asume la responsabilidad total sobre la autoría, veracidad, rigor técnico y defensa de todo el código, modelos y reportes presentados en este trabajo.
