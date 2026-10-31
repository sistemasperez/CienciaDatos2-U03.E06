import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder


def test_autoencoder_output_contract():
    model = build_autoencoder(tf, latent_dim=16)
    inputs = np.zeros((2, 28, 28, 1), dtype="float32")
    result = model(inputs).numpy()
    assert result.shape == (2, 28, 28, 1)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_encoder_latent_contract():
    encoder, _ = build_encoder_decoder(tf, latent_dim=2)
    inputs = np.zeros((3, 28, 28, 1), dtype="float32")
    mean, log_var = encoder(inputs)
    assert mean.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert np.isfinite(mean.numpy()).all()
    assert np.isfinite(log_var.numpy()).all()


def test_decoder_output_contract():
    _, decoder = build_encoder_decoder(tf, latent_dim=2)
    latent_vector = np.zeros((4, 2), dtype="float32")
    result = decoder(latent_vector).numpy()
    assert result.shape == (4, 28, 28, 1)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_latent_interpolation_smoothness():
    _, decoder = build_encoder_decoder(tf, latent_dim=2)
    z1 = np.array([-1.0, -1.0])
    z2 = np.array([1.0, 1.0])
    alphas = np.linspace(0, 1, 5)
    grid = np.array([(1 - a) * z1 + a * z2 for a in alphas], dtype="float32")
    interpolated = decoder(grid).numpy()
    assert interpolated.shape == (5, 28, 28, 1)
    # Consecutive difference must be finite
    diffs = np.linalg.norm((interpolated[1:] - interpolated[:-1]).reshape(4, -1), axis=1)
    assert np.all(diffs >= 0)
