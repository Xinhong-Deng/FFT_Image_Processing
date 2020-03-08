import sys

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


if __name__ == '__main__':
    mode = 1
    img = "moonlanding.png"
    print(sys.argv)
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

    if mode == 1:
        # call mode 1 function
        exit()

    if mode == 2:
        # call mode 2 function
        exit()

    if mode == 3:
        # call mode 3 function
        exit()

    if mode == 4:
        # call mode 4 function
        exit()
