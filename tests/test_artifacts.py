import json
from pathlib import Path
import pytest

tf = pytest.importorskip("tensorflow")

ROOT = Path(__file__).resolve().parents[1]


def test_saved_models_exist_and_load():
    ae_path = ROOT / "models" / "autoencoder.keras"
    dec_path = ROOT / "models" / "decoder.keras"
    assert ae_path.exists(), "El modelo autoencoder.keras debe existir"
    assert dec_path.exists(), "El modelo decoder.keras debe existir"

    ae = tf.keras.models.load_model(ae_path)
    decoder = tf.keras.models.load_model(dec_path)
    assert ae is not None
    assert decoder is not None


def test_metrics_json_integrity():
    metrics_path = ROOT / "reports" / "training_metrics.json"
    assert metrics_path.exists(), "El archivo training_metrics.json debe existir"

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "autoencoder" in data
    assert "vae" in data
    assert data["autoencoder"]["latent_dim"] == 16
    assert data["vae"]["latent_dim"] == 2
    assert data["vae"]["mean_nn_distance_to_train"] > 0
    assert data["vae"]["mean_sample_diversity"] > 0
