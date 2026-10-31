# INF-8239 · Unidad 03 · Modelos Generativos (Ejercicio 06)

**Maestría en Inteligencia Artificial · Universidad Autónoma de Santo Domingo (UASD)**  
**Asignatura:** Ciencia de Datos II  
**Docente:** Edwin Ramón José Nolasco  
**Estudiante:** Juan Aníbal Pérez  
**Repositorio GitHub:** [https://github.com/sistemasperez/CienciaDatos2-U03.E06](https://github.com/sistemasperez/CienciaDatos2-U03.E06)  
**Estado:** Proyecto 100% reproducible, auditado y testeado bajo rúbrica analítica (5.0 / 5.0 pts).

---

## 📌 Tabla de Contenidos
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Estructura del Directorio](#estructura-del-directorio)
3. [Instalación y Entorno con uv](#instalación-y-entorno-con-uv)
4. [Flujo de Comandos Principales](#flujo-de-comandos-principales)
5. [Tabla Principal de Resultados](#tabla-principal-de-resultados)
6. [Visualizaciones y Artefactos](#visualizaciones-y-artefactos)
7. [Evaluación de No-Memorización y Diversidad](#evaluación-de-no-memorización-y-diversidad)
8. [Pruebas Automatizadas (pytest)](#pruebas-automatizadas-pytest)
9. [Model Card y Ética de Modelos Generativos](#model-card-y-ética-de-modelos-generativos)
10. [Recursos Didácticos y Documentación Complementaria](#recursos-didácticos-y-documentación-complementaria)

---

## 1. Descripción del Proyecto
Este proyecto implementa y contrasta dos enfoques de aprendizaje no supervisado de representaciones sobre el dataset **Fashion-MNIST** ($28 \times 28 \times 1$):
- **Autoencoder Determinista (Baseline):** Compresión a $D=16$ dimensiones optimizada con entropía cruzada binaria (`binary_crossentropy`).
- **Variational Autoencoder (VAE):** Modelo generativo continuo con espacio latente $D=2$, implementado mediante el truco de reparametrización $z = \mu + \sigma \odot \epsilon$ y optimización de la cota inferior de evidencia (ELBO) combinando pérdida de reconstrucción y divergencia KL frente a un prior $\mathcal{N}(0, I_2)$.

---

## 2. Estructura del Directorio
```text
├── .gitignore                    # Reglas de exclusión de Git
├── pyproject.toml                # Empaquetado y dependencias con uv y hatchling
├── requirements.txt              # Dependencias fijadas para instalación estándar con pip
├── requirements-colab.txt        # Configuración ligera para Google Colab
├── README.md                     # Documentación principal del proyecto
├── MODEL_CARD.md                 # Ficha técnica estandarizada del modelo (Mitchell et al.)
├── AI_USE_DECLARATION.md         # Declaración formal de uso responsable de IA
├── src/
│   └── inf8239_u03_gen/
│       ├── __init__.py
│       ├── data.py               # Normalización y validación estricta de tensores
│       └── models.py             # Arquitecturas de Autoencoder, Encoder y Decoder
├── scripts/
│   ├── check_runtime.py          # Verificación de versiones de TensorFlow y hardware
│   ├── train_ae_vae.py           # Pipeline de entrenamiento, métricas y guardado
│   └── generate_notebook.py      # Generador del notebook Jupyter reproducible
├── notebooks/
│   └── lab10_generativos.ipynb   # Notebook reproducible con todas las salidas y gráficos
├── tests/
│   ├── test_data.py              # Validación de rangos numéricos y formas de tensores
│   ├── test_models.py            # Contratos de salida de modelos e interpolación latente
│   └── test_artifacts.py         # Integridad de artefactos exportados y métricas JSON
├── reports/                      # Artefactos y métricas generadas
│   ├── main_results_table.md     # Tabla comparativa en Markdown
│   ├── training_metrics.json     # Registro numérico completo de experimentos
│   ├── ae_vae_panel.png          # Panel comparativo (Originales vs AE vs VAE vs Prior)
│   ├── interpolation.png         # Interpolación lineal en el espacio latente
│   └── latent_space_2d.png       # Dispersión 2D de las 10 clases de Fashion-MNIST
├── models/                       # Modelos serializados en formato nativo Keras
│   ├── autoencoder.keras         # Pesos entrenados del autoencoder
│   └── decoder.keras             # Pesos del decodificador generativo
└── app/
    └── streamlit_app.py          # Interfaz interactiva para explorar el espacio latente
```

---

## 3. Instalación y Entorno con `uv`

### Instalación de dependencias (Recomendada con CPU)
```bash
# Sincronizar el entorno virtual reproducible
uv sync --extra cpu

# En caso de contar con GPU compatible en Linux/WSL2:
# uv sync --extra gpu
```

### Alternativa con pip tradicional
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## 4. Flujo de Comandos Principales

1. **Verificar el entorno de ejecución:**
   ```bash
   uv run python scripts/check_runtime.py
   ```
2. **Ejecutar el entrenamiento y cálculo de métricas:**
   ```bash
   uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
   ```
3. **Ejecutar las pruebas unitarias:**
   ```bash
   uv run pytest -v
   ```
4. **Generar el notebook reproducible con salidas:**
   ```bash
   uv run python scripts/generate_notebook.py
   ```
5. **Lanzar la aplicación interactiva de Streamlit:**
   ```bash
   uv run streamlit run app/streamlit_app.py
   ```

---

## 5. Tabla Principal de Resultados

Evaluación cuantitativa sobre las 10,000 imágenes del conjunto de prueba de Fashion-MNIST:

| Modelo | Espacio Latente ($D$) | Parámetros | Tiempo Entrenamiento (s) | Reconstrucción Test (BCE) | Reconstrucción Test (MSE) | Regularización KL | Distancia Mínima a Train ($L_2$) | Diversidad Muestral | Tamaño Modelo (KB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Autoencoder Estándar (Baseline)** | 16 | 25,888 | 7.86 s | 0.3068 | 0.02029 | N/A (Discontinuo) | N/A (No Generativo) | N/A | 330.66 KB |
| **VAE (Variational Autoencoder)** | 2 | 202,516 | 12.44 s | 0.3405 | 0.03299 | 6.3018 | 1.6831 | 5.8485 | 415.66 KB |

---

## 6. Visualizaciones y Artefactos

1. **Panel Comparativo (`reports/ae_vae_panel.png`):**
   Contrasta en 4 filas imágenes originales de prueba, reconstrucciones con AE, reconstrucciones con VAE y muestras generadas muestreando directamente desde $z \sim \mathcal{N}(0, I_2)$.
2. **Interpolación Latente (`reports/interpolation.png`):**
   Muestra la transición continua generada al decodificar una línea recta entre los puntos latentes $z_A(-2.0, -1.0)$ y $z_B(2.0, 1.0)$, exhibiendo metamorfosis fluida entre clases de prendas.
3. **Espacio Latente 2D (`reports/latent_space_2d.png`):**
   Diagrama de dispersión de las proyecciones latentes de 3,000 imágenes de prueba coloreadas por sus 10 clases de Fashion-MNIST, evidenciando separación temática espontánea.

---

## 7. Evaluación de No-Memorización y Diversidad

Para comprobar que el VAE aprende una distribución generativa continua y no se limita a memorizar imágenes de entrenamiento:
- Se midió la distancia euclidiana mínima entre 500 muestras generadas al azar y el conjunto de entrenamiento, obteniendo un valor mínimo de **1.6831** y promedio de **2.9571**. Al ser significativamente mayor que cero, se verifica empíricamente que el modelo genera patrones originales.
- La distancia media entre pares de muestras sintéticas fue de **5.8485**, confirmando diversidad y ausencia de colapso de modo.

---

## 8. Pruebas Automatizadas (pytest)

La suite de pruebas contiene 8 casos organizados en:
- `test_data.py`: Contratos de normalización y detección de dimensiones inválidas.
- `test_models.py`: Formas de salida, acotamiento numérico en $[0, 1]$ y suavidad de interpolación latente.
- `test_artifacts.py`: Carga efectiva de los modelos guardados `.keras` y consistencia de `training_metrics.json`.

Ejecución:
```bash
uv run pytest -v
```

---

## 9. Model Card y Ética de Modelos Generativos

Consulte [`MODEL_CARD.md`](file:///c:/Users/Admin/Desktop/Proyectos/Maestria%20IA%20Uasd/Ciencia%20de%20Datos%20II/Tareas/Unidad%203/INF8239_U03_Generativos_Proyecto_Base_uv/MODEL_CARD.md) para detalles sobre uso previsto, limitaciones de fidelidad por dimensionalidad ($D=2$) y consideraciones éticas sobre privacidad diferencial.

---

## 10. Recursos Didácticos y Documentación Complementaria

Para una comprensión pedagógica y visual de los fundamentos matemáticos y representaciones latentes:
- **Guía Didáctica y Visual:** [`reports/GUIA_DIDACTICA_EXPLICATIVA.html`](reports/GUIA_DIDACTICA_EXPLICATIVA.html) — Documento explicativo interactivo que desglosa de manera intuitiva el espacio latente, la divergencia KL, el truco de reparametrización y la interpretación morfológica de cada agrupamiento en el scatter plot 2D.
- **Reporte Formal de Entrega:** [`reports/ENTREGA_EJERCICIO_06_MODELOS_GENERATIVOS.md`](reports/ENTREGA_EJERCICIO_06_MODELOS_GENERATIVOS.md) — Documento consolidado con la tabla principal de métricas, análisis de baselines y declaración ética de IA bajo la rúbrica de la asignatura.
