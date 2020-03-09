import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.fftpack as sf


def pad_to_power_2(x_raw):
    N = x_raw.shape[0]
    N_ceil = pow(2, math.ceil(math.log(N, 2)))
    if (N - N_ceil == 0):
        return x_raw, N
    else:
        x_append = np.zeros(N_ceil - N)
        return np.append(x_raw, x_append), N_ceil


def dft(x):
    x = np.asarray(x, dtype=np.complex_)
    N = x.shape[0]
    n_small = np.asarray(range(0, N), dtype=np.complex_)
    k = n_small.reshape((N, 1))
    coe = np.exp(-2j * np.pi * k * n_small / N)
    return np.dot(coe, x)


def dft_inv(X):
    X = np.asarray(X, dtype=np.complex_)
    N = X.shape[0]
    n_small = np.asarray(range(0, N), dtype=np.complex_)
    k = n_small.reshape((N, 1))
    coe = np.exp(2j * np.pi * k * n_small / N)
    return  np.dot(coe, X)


def fft(x_raw):
    x_raw = np.asarray(x_raw)
    x, N = pad_to_power_2(x_raw)

    if N <= 2:
        return dft(x)
    else:
        X_even =  fft(x[::2])
        X_odd =  fft(x[1::2])
        coe = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate((X_even + coe[:int(N / 2)] * X_odd,
                               X_even + coe[int(N / 2):] * X_odd))


def fft_inv_helper(x_raw):
    x_raw = np.asarray(x_raw)
    x, N = pad_to_power_2(x_raw)

    if N <= 2:
        return dft_inv(x)
    else:
        X_even =fft_inv_helper(x[::2])
        X_odd =fft_inv_helper(x[1::2])
        coe = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + coe[:int(N / 2)] * X_odd,
                               X_even + coe[int(N / 2):] * X_odd])


def fft_inv(x_raw):
    x, N = pad_to_power_2(x_raw)
    coe = 1/N
    result = coe* np.array(fft_inv_helper(x_raw))
    return result


def twodfft(signal: np.array):
    row, column = signal.shape
    columns_transformation = []
    for c in range(column):
        columns_transformation.append(fft(signal[:, c]))

    final_result = []
    columns_transformation = np.asarray(columns_transformation, dtype=np.complex_).T
    for r in range(row):
        final_result.append(fft(columns_transformation[r, :]))

    final_result = np.asarray(final_result, dtype=np.complex_)
    zero_padding = np.zeros((columns_transformation.shape[0] - row, final_result.shape[1]), dtype=np.complex_)
    return np.append(final_result, zero_padding, 0)


def twodfft_inverse(signal: np.array):
    row, column = signal.shape
    columns_inverse = []
    for c in range(column):
        columns_inverse.append(fft_inv(signal[:, c]))

    final_result = []
    columns_inverse = np.asarray(columns_inverse, dtype=np.complex_).T
    for r in range(row):
        final_result.append(sf.ifft(columns_inverse[r, :]))

    final_result = np.asarray(final_result, dtype=np.complex_)
    zero_padding = np.zeros((columns_inverse.shape[0] - row, final_result.shape[1]), dtype=np.complex_)
    return np.append(final_result, zero_padding, 0)


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
    img_data = np.append(img_data, np.zeros((512 - 474, 630), dtype=np.complex_), axis=0)
    img_data = np.append(img_data, np.zeros((512, 1024 - 630), dtype=np.complex_), axis=1)

    sfft = sf.fft2(img_data)
    # sifft = sf.ifft2(sfft)
    # d = twodfft(img_data)
    # id = twodfft_inverse(d)

    c0 = fft_inv(sfft[:, 0])
    sifft = sf.ifft(sfft[:, 0])

    # data = [-1.99765739e+04-6.09136962e+04j , 5.62322155e+03+1.29968394e+04j, 1.42919663e+03+2.40975589e+03j, -2.49200697e+03+2.08932389e+03j]
    data = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
    ft = fft(data)
    ift = fft_inv(ft)
    print(ift)
    # sfft = sf.fft(data)
    # sift = sf.ifft(sfft)

    print("1")

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
