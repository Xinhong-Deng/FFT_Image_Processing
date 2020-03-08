import numpy as np


def twodfft(signal: np.array):
    row, column = signal.shape
    columns_transformation = []
    for c in range(column):
        columns_transformation.append(fft(signal[:, c]))
    return fft(columns_transformation)


def twodfft_inverse(signal: np.array):
    row, column = signal.shape
    columns_inverse = []
    for c in range(column):
        columns_inverse.append(fft_inverse(signal[:, c]))
    return 1/(row * column) * fft_inverse(columns_inverse)

