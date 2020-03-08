import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.fftpack as sf

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n_small = np.asarray(range(0,N))
    k = n_small.reshape((N, 1))
    coe = np.exp(-2j * np.pi * k * n_small / N)
    return np.dot(coe, x)


def dft_inv(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    n_small = np.asarray(range(0, N))
    k = n_small.reshape((N, 1))
    coe = np.exp(2j * np.pi * k * n_small / N)
    return (1/N) * np.dot(coe, X)



def fft(x_raw):
    x_raw = np.asarray(x_raw, dtype=float)
    N_raw = x_raw.shape[0]
    N = pow(2, math.floor(math.log(N_raw, 2)))
    x = x_raw[:N]
    if N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        coe = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate((X_even + coe[:int(N / 2)] * X_odd,
                               X_even + coe[int(N / 2):] * X_odd))



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
        coe = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + coe[:int(N / 2)] * X_odd,
                               X_even + coe[int(N / 2):] * X_odd])


def twodfft(signal: np.array):
    row, column = signal.shape
    columns_transformation = np.zeros((row, column))
    for c in range(column):
        print(signal[:, c].shape)
        columns_transformation[:, c] = fft(signal[:, c]).T

    final_result = np.zeros((row, column))
    for r in range(row):
        final_result[r, :] = fft(columns_transformation[r, :])
    return final_result


def twodfft_inverse(signal: np.array):
    row, column = signal.shape
    columns_inverse = np.zeros(row, column)
    for c in range(column):
        columns_inverse[:, c] = fft_inv(signal[:, c]).T

    final_result = np.zeros(row, column)
    for r in range(row):
        final_result[r, :] = fft_inv(columns_inverse[r, :])
    return 1/(row * column) * final_result


if __name__ == '__main__':
    mode = 1
    img = "moonlanding.png"
    index = 1
    while index < len(sys.argv):
        if sys.argv[index] == "-m":
            index += 1
            mode = sys.argv[index]
        elif sys.argv[index] == "-i":
            index += 1
            img = sys.argv[index]
        else:
            print("error in argument")
            exit()

        index += 1

    img_data = plt.imread(img).astype(float)

    sfft = sf.fft(img_data)
    d = twodfft(img_data)

    if all(sfft == d):
        print("tick!")
    
    # if mode == 1:
    #     # call mode 1 function
    #     exit()
    #
    # if mode == 2:
    #     # call mode 2 function
    #     exit()
    #
    # if mode == 3:
    #     # call mode 3 function
    #     exit()
    #
    # if mode == 4:
    #     # call mode 4 function
    #     exit()
