| Modelo | Espacio Latente ($D$) | Parámetros | Tiempo Entrenamiento (s) | Reconstrucción Test (BCE) | Reconstrucción Test (MSE) | Regularización KL | Distancia Mínima a Train (L2) | Diversidad Muestral | Tamaño Modelo (KB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Autoencoder Estándar (Baseline)** | 16 | 25,888 | 7.86 s | 0.3068 | 0.02029 | N/A (Espacio Discontinuo) | N/A (No Generativo) | N/A | 330.66 KB |
| **VAE (Variational Autoencoder)** | 2 | 202,516 | 12.44 s | 0.3405 | 0.03299 | 6.3018 | 1.6831 | 5.8485 | 415.66 KB |
