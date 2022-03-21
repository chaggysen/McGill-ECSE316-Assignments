import numpy as np
import matplotlib.pyplot as plt
import cv2

class FOURIER_TRANSFORM:
    def __init__(self, args):
        self.mode = args.MODE
        self.image = cv2.imread(args.IMAGE, cv2.IMREAD_GRAYSCALE)
        self.test()

    def dft_naive_1d(self, x):
        # Computes DFT of 1D array x
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    def dft_naive_2d(self, x):
        # Computes DFT of 2D array x
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        M = x.shape[1]
        res = np.zeros((N, M), dtype=complex)
        for k in range(N):
            for l in range(M):
                res[k, l] = self.dft_naive_1d(x[k, :])[l]
        return res

    def display(self, image):
        # Display the image
        plt.imshow(image, cmap='gray')
        plt.show()

    def test(self):
        print(np.allclose(self.dft_naive_1d([1,2,3,4,5,6]), np.fft.fft([1,2,3,4,5,6])))
        print(np.allclose(self.dft_naive_2d([[1,2],[3,4],[5,6]]), np.fft.fft([[1,2],[3,4],[5,6]])))