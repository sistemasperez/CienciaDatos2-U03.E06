from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            mean, log_variance = self.encoder(images, training=True)
            epsilon = tf.random.normal(tf.shape(mean))
            latent = mean + tf.exp(0.5 * log_variance) * epsilon
            reconstruction = self.decoder(latent, training=True)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1))
            total_loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def image_panel(path: Path, originals, ae_reconstructed, vae_reconstructed, generated):
    figure, axes = plt.subplots(4, 12, figsize=(14, 5))
    row_labels = ["Originales", "AE (Reconstrucción)", "VAE (Reconstrucción)", "VAE (Muestras Prior)"]
    for column in range(12):
        for row, collection in enumerate([originals, ae_reconstructed, vae_reconstructed, generated]):
            axes[row, column].imshow(collection[column].squeeze(), cmap="gray")
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_title(row_labels[row], fontsize=9, loc="left", pad=4)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_latent_space_2d(path: Path, encoder, x_test, y_test):
    mean, _ = encoder.predict(x_test[:3000], batch_size=512, verbose=0)
    figure, ax = plt.subplots(figsize=(8, 6))
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    scatter = ax.scatter(mean[:, 0], mean[:, 1], c=y_test[:3000], cmap="tab10", alpha=0.6, s=12)
    ax.set_title("Espacio Latente 2D del VAE (Fashion-MNIST Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimensión Latente $z_1$", fontsize=10)
    ax.set_ylabel("Dimensión Latente $z_2$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    colorbar = figure.colorbar(scatter, ax=ax, ticks=range(10))
    colorbar.ax.set_yticklabels(class_names)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_memorization_and_diversity(decoder, x_train, n_samples: int = 500):
    rng = np.random.default_rng(SEED)
    z_samples = rng.normal(size=(n_samples, 2))
    generated = decoder.predict(z_samples, batch_size=256, verbose=0).reshape(n_samples, -1)
    
    # Train subset for comparison
    n_train_sub = min(5000, len(x_train))
    x_train_flat = x_train[:n_train_sub].reshape(n_train_sub, -1)
    
    # Nearest neighbor from generated to training set
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(x_train_flat)
    distances_to_train, _ = nn.kneighbors(generated)
    mean_nn_dist = float(np.mean(distances_to_train))
    min_nn_dist = float(np.min(distances_to_train))
    
    # Diversity: pairwise distances among generated samples
    pairs_idx_a = rng.choice(n_samples, size=1000, replace=True)
    pairs_idx_b = rng.choice(n_samples, size=1000, replace=True)
    mask = pairs_idx_a != pairs_idx_b
    pairwise_dists = np.linalg.norm(generated[pairs_idx_a[mask]] - generated[pairs_idx_b[mask]], axis=1)
    mean_diversity = float(np.mean(pairwise_dists))
    
    return {
        "mean_nn_distance_to_train": round(mean_nn_dist, 4),
        "min_nn_distance_to_train": round(min_nn_dist, 4),
        "mean_sample_diversity": round(mean_diversity, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-ae", type=int, default=6)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    reports = ROOT / "reports"
    models = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    print("Cargando y validando Fashion-MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    if args.limit:
        x_train = x_train[: args.limit]
        x_test = x_test[: max(100, args.limit // 5)]
        y_train = y_train[: args.limit]
        y_test = y_test[: max(100, args.limit // 5)]
    validate_images(x_train)

    # 1. Entrenamiento Autoencoder
    print(f"\n--- Entrenando Autoencoder ({args.epochs_ae} épocas) ---")
    start = perf_counter()
    autoencoder = build_autoencoder(tf, latent_dim=16)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(
        x_train, x_train,
        validation_split=0.1,
        epochs=args.epochs_ae,
        batch_size=256,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=2
    )
    ae_seconds = perf_counter() - start

    # 2. Entrenamiento VAE
    print(f"\n--- Entrenando VAE ({args.epochs_vae} épocas) ---")
    encoder, decoder = build_encoder_decoder(tf, latent_dim=2)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")
    start = perf_counter()
    vae.fit(x_train, epochs=args.epochs_vae, batch_size=256, verbose=2)
    vae_seconds = perf_counter() - start

    # 3. Generación de Paneles e Interpolaciones
    print("\nGenerando visualizaciones y paneles...")
    originals = x_test[:12]
    ae_reconstructed = autoencoder.predict(originals, verbose=0)
    mean, log_variance = encoder.predict(originals, verbose=0)
    vae_reconstructed = decoder.predict(mean, verbose=0)
    generated = decoder.predict(np.random.default_rng(SEED).normal(size=(12, 2)), verbose=0)

    image_panel(reports / "ae_vae_panel.png", originals, ae_reconstructed, vae_reconstructed, generated)

    # Interpolación latente entre dos puntos extremos
    grid = np.array([(1 - value) * np.array([-2.0, -1.0]) + value * np.array([2.0, 1.0]) for value in np.linspace(0, 1, 12)])
    interpolation = decoder.predict(grid, verbose=0)
    figure, axes = plt.subplots(1, 12, figsize=(14, 1.6))
    for axis, image in zip(axes, interpolation):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.axis("off")
    figure.suptitle("Interpolación Latente Lineal $z_A(-2.0, -1.0) \\rightarrow z_B(2.0, 1.0)$", fontsize=11)
    figure.tight_layout()
    figure.savefig(reports / "interpolation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Dispersión del espacio latente 2D
    plot_latent_space_2d(reports / "latent_space_2d.png", encoder, x_test, y_test)

    # 4. Guardar modelos
    autoencoder_path = models / "autoencoder.keras"
    decoder_path = models / "decoder.keras"
    autoencoder.save(autoencoder_path)
    decoder.save(decoder_path)

    # 5. Métricas cuantitativas en Test Set
    print("\nCalculando métricas cuantitativas sobre el test set...")
    ae_test_preds = autoencoder.predict(x_test, batch_size=512, verbose=0)
    ae_test_mse = float(np.mean((x_test - ae_test_preds) ** 2))
    ae_test_bce = float(tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_test, ae_test_preds)).numpy())

    vae_mean_test, vae_logvar_test = encoder.predict(x_test, batch_size=512, verbose=0)
    vae_test_preds = decoder.predict(vae_mean_test, batch_size=512, verbose=0)
    vae_test_mse = float(np.mean((x_test - vae_test_preds) ** 2))
    vae_test_bce = float(tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_test, vae_test_preds)).numpy())
    vae_test_kl = float(-0.5 * np.mean(np.sum(1 + vae_logvar_test - np.square(vae_mean_test) - np.exp(vae_logvar_test), axis=1)))

    # Memorización y diversidad
    mem_div_metrics = evaluate_memorization_and_diversity(decoder, x_train, n_samples=500)

    ae_size_kb = round(autoencoder_path.stat().st_size / 1024, 2)
    decoder_size_kb = round(decoder_path.stat().st_size / 1024, 2)

    metrics = {
        "dataset": "Fashion-MNIST",
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "autoencoder": {
            "latent_dim": 16,
            "parameters": int(autoencoder.count_params()),
            "train_seconds": round(ae_seconds, 2),
            "test_bce_loss": round(ae_test_bce, 4),
            "test_mse": round(ae_test_mse, 5),
            "file_size_kb": ae_size_kb,
        },
        "vae": {
            "latent_dim": 2,
            "encoder_parameters": int(encoder.count_params()),
            "decoder_parameters": int(decoder.count_params()),
            "total_parameters": int(encoder.count_params() + decoder.count_params()),
            "train_seconds": round(vae_seconds, 2),
            "test_reconstruction_bce": round(vae_test_bce, 4),
            "test_reconstruction_mse": round(vae_test_mse, 5),
            "test_kl_divergence": round(vae_test_kl, 4),
            "file_size_kb": decoder_size_kb,
            **mem_div_metrics
        }
    }

    (reports / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Tabla principal en Markdown
    table_md = f"""| Modelo | Espacio Latente ($D$) | Parámetros | Tiempo Entrenamiento (s) | Reconstrucción Test (BCE) | Reconstrucción Test (MSE) | Regularización KL | Distancia Mínima a Train (L2) | Diversidad Muestral | Tamaño Modelo (KB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Autoencoder Estándar (Baseline)** | 16 | {autoencoder.count_params():,} | {ae_seconds:.2f} s | {ae_test_bce:.4f} | {ae_test_mse:.5f} | N/A (Espacio Discontinuo) | N/A (No Generativo) | N/A | {ae_size_kb} KB |
| **VAE (Variational Autoencoder)** | 2 | {encoder.count_params() + decoder.count_params():,} | {vae_seconds:.2f} s | {vae_test_bce:.4f} | {vae_test_mse:.5f} | {vae_test_kl:.4f} | {mem_div_metrics['min_nn_distance_to_train']:.4f} | {mem_div_metrics['mean_sample_diversity']:.4f} | {decoder_size_kb} KB |
"""
    (reports / "main_results_table.md").write_text(table_md, encoding="utf-8")

    print("\nResultados guardados exitosamente en 'reports/':")
    print(table_md)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
