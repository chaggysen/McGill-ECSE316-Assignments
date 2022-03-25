import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm

class FOURIER_TRANSFORM:
    def __init__(self, args):
        self.mode = args.MODE
        self.image = cv2.imread(args.IMAGE, cv2.IMREAD_GRAYSCALE)
        self.test()
        if(self.mode == 1):
            self.perform_fft(self.image)
        elif(self.mode == 2):
            self.perform_denoising(self.image)
        elif(self.mode == 3):
            self.perform_compression(self.image)
        elif(self.mode == 4):
            self.perform_runtime_analysis()

    def perform_fft(self, image):
        pass

    def perform_denoising(self, image):
        pass

    def perform_compression(self, image):
        pass

    def perform_runtime_analysis(self):
        pass

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

    def dft_inverse_1d(self, X):
        X = np.asarray(X, dtype=complex)
        N = X.shape[0]
        x = np.zeros(N, dtype=complex)
        # sum from n=0 to N-1 of 1/N*X[k]*exp(i*2*pi/N*k*n) where k=0,1,...,N-1
        for n in range(N):
            for k in range(N):
                x[n] += 1/N*X[k]*np.exp(2j*np.pi/N*k*n)
        return x

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

    def dft_inverse_2d(self, F):
        F = np.asarray(F, dtype=complex)
        M = F.shape[0]
        N = F.shape[1]
        f = np.zeros((M, N), dtype=complex)
        # sum from l=0 to N=1 of sum from k=0 to M=1 of 1/(N*M)*F[k,l]*exp(i*2*pi/M*k*m)*exp(i*2*pi/N*l*n) for m=0,1,...,M-1 and n=0,1,...,N-1
        for l in range(N):
            for k in range(M):
                for m in range(M):
                    for n in range(N):
                        f[m,n] += 1/(N*M)*F[k,l]*np.exp(2j*np.pi/M*k*m)*np.exp(2j*np.pi/N*l*n)
        return f

    def display(self, image):
        # Display the image
        plt.imshow(image, cmap='gray', norm=LogNorm())
        plt.show()

    def test(self):
        print(np.allclose(self.dft_naive_1d([1,2,3,4,5,6]), np.fft.fft([1,2,3,4,5,6])))
        print(np.allclose(self.dft_naive_2d([[1,2],[3,4],[5,6]]), np.fft.fft2([[1,2],[3,4],[5,6]])))
        print(np.allclose(self.dft_inverse_1d([1,2,3,4,5,6]), np.fft.ifft([1,2,3,4,5,6])))
        print(np.allclose(self.dft_inverse_2d([[1,2],[3,4],[5,6]]), np.fft.ifft2([[1,2],[3,4],[5,6]])))
        #self.display(abs(np.fft.fft2(self.image)))