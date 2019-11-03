import numpy as np
import audio_utils


def test_root_mean_square():
    sine_wave = np.sin(np.linspace(-np.pi, np.pi, 201))
    assert audio_utils.root_mean_square(sine_wave) == 0.7053456158585982
