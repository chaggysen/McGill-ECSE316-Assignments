import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

class FOURIER_TRANSFORM:
    def __init__(self, args):
        self.mode = args.MODE
        self.image = cv2.imread(args.IMAGE, cv2.IMREAD_GRAYSCALE)
        self.test()

    def dft_naive(self, x):
        # Computes DFT of 1D array x
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    def dft_naive_1d(self, x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        X = np.zeros(N, dtype=complex)
        # sum from n=0 to N-1 of x[n]*exp(-i*2*pi/N*k*n) where k=0,1,...,N-1
        for n in range(N):
            for k in range(N):
                X[k] += x[n]*np.exp(-2j*np.pi/N*k*n)
        return X

    def dft_naive_2d(self, f):
        f = np.asarray(f, dtype=complex)
        M = f.shape[0]
        N = f.shape[1]
        F = np.zeros((M, N), dtype=complex)
        #F[k,l] = sum from n=0 to N-1 of sum from m=0 to M-1 of f[m,n]*exp(-i*2*pi/M*k*m)*exp(-i*2*pi/N*l*n) for k=0,1,...,M-1 and l=0,1,...,N-1
        for n in range(N):
            for m in range(M):
                for k in range(M):
                    for l in range(N):
                        F[k,l] += f[m,n]*np.exp(-2j*np.pi/M*k*m)*np.exp(-2j*np.pi/N*l*n)
        return F

    def display(self, image):
        # Display the image
        plt.imshow(image, cmap='gray')
        plt.show()

    def test(self):
        print(np.allclose(self.dft_naive_1d([1,2,3,4,5,6]), np.fft.fft([1,2,3,4,5,6])))
        print(np.allclose(self.dft_naive_2d([[1,2],[3,4],[5,6]]), np.fft.fft2([[1,2],[3,4],[5,6]])))
        #self.display(20*np.log10(abs(np.fft.fft2(self.image))))