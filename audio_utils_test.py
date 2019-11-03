import numpy as np
import audio_utils


def test_root_mean_square():
    sine_wave = np.sin(np.linspace(-np.pi, np.pi, 201))
    c = audio_utils.root_mean_square(sine_wave)
    assert 0.7 < c < .71
