import argparse
from typing import Tuple

import numpy as np
from scipy.io import wavfile
import sounddevice as sd

from pvals.pvals import PVALS, Initializer


def arg_parser() -> Tuple[float, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise", default=1, type=float, help="Amount of noise to add to the tensor"
    )
    parser.add_argument(
        "--initializer",
        default="gaussian",
        type=str,
        help="Distribution to initialize the ALS algorithm",
    )

    parser.add_argument("--plot_loss", dest="plot_loss", action="store_true")
    parser.add_argument("--dont_plot_loss", dest="plot_loss", action="store_false")
    parser.set_defaults(plot_loss=True)

    args = parser.parse_args()
    return args.noise, args.initializer, args.plot_loss


def main() -> None:
    noise, initializer, plot_loss = arg_parser()

    samplerate0, signal0 = wavfile.read("data/Speech1.wav")
    samplerate1, signal1 = wavfile.read("data/Speech2.wav")

    # Make s0 and s1 have the same length
    signal1 = signal1[: len(signal0)]

    sd.play(signal0, samplerate0)
    sd.wait()

    sd.play(signal1, samplerate1)
    sd.wait()

    # Mix the signals
    rng = np.random.default_rng(17)
    A = rng.normal(size=(2, 2))
    mixed_signals = A @ np.stack((signal0, signal1))

    sd.play(mixed_signals[0, :], samplerate0)
    sd.wait()

    # Construct tensor
    X = np.zeros(shape=(2, 2, 50))
    segments = np.linspace(start=0, stop=len(signal0), num=51, dtype=int)
    for i in range(50):
        seg = mixed_signals[:, segments[i] : segments[i + 1]]
        X[:, :, i] = (1 / segments[1]) * seg @ seg.T

    # The noise needs to be on a similar scale as the data
    mean_val = X.mean()
    X_noise = X + (noise * mean_val) * rng.normal(size=X.shape)

    # Decompose the tensor
    if initializer == "gaussian":
        als = PVALS(X_noise, r=2, initializer=Initializer.GAUSSIAN)
    else:
        als = PVALS(X_noise, r=2, initializer=Initializer.UNIFORM)

    Ahat, _, _ = als.pvals()

    # Plot the loss curve if the user requested (will do this by default)
    if plot_loss:
        als.plot_losses()

    Ahat_inv = np.linalg.inv(Ahat)
    separated = Ahat_inv @ mixed_signals

    sd.play(separated[0, :], samplerate0)
    sd.wait()

    sd.play(separated[1, :], samplerate1)
    sd.wait()


if __name__ == "__main__":
    main()
