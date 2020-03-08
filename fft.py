import numpy as np
import math

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n_small = np.asarray(range(0,N))
    k = np.transpose(n_small)
    power = np.exp(-2j * np.pi * k * n_small / N)
    return np.dot(power, x)


def dft_inv(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    n_small = np.asarray(range(0, N))
    k = np.transpose(n_small)
    power = np.exp(2j * np.pi * k * n_small / N)
    return (1/N) * np.dot(power, X)



def fft(x_raw):
    x_raw = np.asarray(x_raw, dtype=float)
    N_raw = x_raw.shape[0]
    N = pow(2, math.floor(math.log(N_raw, 2)))
    x = x_raw[:,:N]
    if N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        coe = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + coe[:int(N / 2)] * X_odd,
                               X_even + coe[int(N / 2):] * X_odd])



def fft_inv(x_raw):
    x_raw = np.asarray(x_raw, dtype=float)
    N_raw = x_raw.shape[0]
    N = pow(2, math.floor(math.log(N_raw, 2)))
    x = x_raw[:, :N]
    if N <= 2:
        return dft_inv(x)
    else:
        X_even = fft_inv(x[::2])
        X_odd = fft_inv(x[1::2])
        terms = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N / 2)] * X_odd,
                               X_even + terms[int(N / 2):] * X_odd])