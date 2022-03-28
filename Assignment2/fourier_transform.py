from time import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm
from utils import Fourier


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

    def display(self, image):
        # Display the image
        plt.imshow(image, cmap='gray', norm=LogNorm())
        plt.show()

    def closest_power_of_2(self, n):
        # Find the closest power of 2
        return 2**(n-1).bit_length()

    def perform_fft(self, image):
        # Output a one by two subplot where the first subplot is the original image and the second subplot is the denoised image
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray', norm=LogNorm())
        plt.title('Original Image')

        # Before doing the FFT, we need to pad the image so that it is a power of 2
        # The padding is done by adding zeros to the end of the image
        N = image.shape[0]
        M = image.shape[1]
        N_padded = self.closest_power_of_2(N)
        M_padded = self.closest_power_of_2(M)
        # pad the image with zeros
        padded_image = np.zeros((N_padded, M_padded), dtype=complex)
        padded_image[:N, :M] = image

        # Perform the FFT of the image and set all the high frequency components to zero
        transformed_image = Fourier.fft_2d(padded_image)

        # Perform the inverse FFT of the image
        plt.subplot(1, 2, 2)
        plt.imshow(abs(transformed_image), norm=LogNorm())
        plt.title('Default Image')
        plt.show()

    def perform_denoising(self, image):
        # Output a one by two subplot where the first subplot is the original image and the second subplot is the denoised image
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray', norm=LogNorm())
        plt.title('Original Image')

        # Before doing the FFT, we need to pad the image so that it is a power of 2
        # The padding is done by adding zeros to the end of the image
        N = image.shape[0]
        M = image.shape[1]
        N_padded = self.closest_power_of_2(N)
        M_padded = self.closest_power_of_2(M)
        # pad the image with zeros
        padded_image = np.zeros((N_padded, M_padded), dtype=complex)
        padded_image[:N, :M] = image

        # Perform the FFT of the image and set all the high frequency components to zero
        transformed_image = Fourier.fft_2d(padded_image)
        fraction = 0.1
        r, c = transformed_image.shape
        transformed_image[int(r*fraction):int(r*(1-fraction))] = 0
        transformed_image[:, int(c*fraction):int(c*(1-fraction))] = 0

        # Perform the inverse FFT of the image
        denoised_image = Fourier.fft_inverse_2d(transformed_image)
        denoised_image = denoised_image[:N, :M]
        plt.subplot(1, 2, 2)
        plt.imshow(abs(denoised_image), cmap='gray', norm=LogNorm())
        plt.title('Denoised Image')
        plt.show()

    def perform_compression(self, image):
        pass

    def perform_runtime_analysis(self):
        pass

    def test(self):
        print(np.allclose(Fourier.dft_naive_1d(
            [1, 2, 3, 4, 5, 6]), np.fft.fft([1, 2, 3, 4, 5, 6])))
        print(np.allclose(Fourier.dft_naive_2d(
            [[1, 2], [3, 4], [5, 6]]), np.fft.fft2([[1, 2], [3, 4], [5, 6]])))
        print(np.allclose(Fourier.dft_inverse_1d(
            [1, 2, 3, 4, 5, 6]), np.fft.ifft([1, 2, 3, 4, 5, 6])))
        print(np.allclose(Fourier.dft_inverse_2d(
            [[1, 2], [3, 4], [5, 6]]), np.fft.ifft2([[1, 2], [3, 4], [5, 6]])))
        x = np.random.rand(64)
        print(np.allclose(Fourier.fft_1d(x), np.fft.fft(x)))
        print(np.allclose(Fourier.fft_inverse_1d(x), np.fft.ifft(x)))
        y = np.random.rand(64, 64)
        print(np.allclose(Fourier.fft_2d(y), np.fft.fft2(y)))
        print(np.allclose(Fourier.fft_inverse_2d(y), np.fft.ifft2(y)))
        # self.display(abs(np.fft.fft2(self.image)))
