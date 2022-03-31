from time import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm
from utils import *


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

        # Perform the FFT of the image
        transformed_image = fft_2d(padded_image)

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
        transformed_image = fft_2d(padded_image)
        fraction = 0.1
        print("Fraction used: ", 100*fraction, "%")
        r, c = transformed_image.shape
        transformed_image[int(r*fraction):int(r*(1-fraction))] = 0
        transformed_image[:, int(c*fraction):int(c*(1-fraction))] = 0

        # Print the number of non-zeros you are using and the fraction they represent of the original Fourier coefficients
        print("Number of non-zeros before converting back to original size: ",
              np.count_nonzero(transformed_image))
        print("Fraction before converting back to original size: ",
              100*np.count_nonzero(transformed_image)/(r*c), "%")

        # Perform the inverse FFT of the image
        denoised_image = fft_inverse_2d(transformed_image)
        denoised_image = denoised_image[:N, :M]
        print("Number of non-zeros after converting back to original size: ",
              np.count_nonzero(denoised_image))
        print("Fraction after converting back to original size: ",
              100*np.count_nonzero(denoised_image)/(N*M), "%")
        plt.subplot(1, 2, 2)
        plt.imshow(abs(denoised_image), cmap='gray', norm=LogNorm())
        plt.title('Denoised Image')
        plt.show()

    def perform_compression(self, image):
        # Before doing the FFT, we need to pad the image so that it is a power of 2
        # The padding is done by adding zeros to the end of the image
        N = image.shape[0]
        M = image.shape[1]
        N_padded = self.closest_power_of_2(N)
        M_padded = self.closest_power_of_2(M)
        # pad the image with zeros
        padded_image = np.zeros((N_padded, M_padded), dtype=complex)
        padded_image[:N, :M] = image
        count = N * M
        compression_lvls = [0, 20, 40, 65, 80, 99.9]

        # Perform the FFT of the image
        transformed_image = fft_2d(padded_image)

        compressed_images = []
        for cl in compression_lvls:
            compressed_images.append(
                self.image_compression(transformed_image, cl, count))

        # render images
        img_counter, r, c = 0, 2, 3
        fig, ax = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                ax[i, j].set_title(
                    'Current compression level:{}%'.format(compression_lvls[img_counter]))
                ax[i, j].imshow(np.real(compressed_images[img_counter])[
                                :N, :M], plt.cm.gray)
                img_counter += 1

        fig.suptitle('Mode 3')
        plt.show()

    def perform_runtime_analysis(self):
        naive_times = {}
        fft_times = {}
        for i in range(5, 9):
            for j in range(10):
                N = 2**i
                M = N
                image = np.random.rand(N, M)
                start = time()
                dft_naive_2d(image)
                end = time()
                if i not in naive_times:
                    naive_times[i] = []
                naive_times[i].append(end - start)
                start = time()
                fft_2d(image)
                end = time()
                if i not in fft_times:
                    fft_times[i] = []
                fft_times[i].append(end - start)

        xval = []
        naive_val = []
        fft_val = []
        yerr1 = []
        yerr2 = []

        print("Naive method:")
        print("")
        for i in naive_times:
            print("Size: ", 2**i, "x", 2**i)
            xval.append("2^" + str(i))
            print("Mean: ", np.mean(naive_times[i]))
            naive_val.append(np.mean(naive_times[i]))
            print("Standard deviation: ", np.std(naive_times[i]))
            yerr1.append(2*np.std(naive_times[i]))
            print("")

        print("FFT method:")
        print("")
        for i in fft_times:
            print("Size: ", 2**i, "x", 2**i)
            print("Mean: ", np.mean(fft_times[i]))
            fft_val.append(np.mean(fft_times[i]))
            print("Standard deviation: ", np.std(fft_times[i]))
            yerr2.append(2*np.std(fft_times[i]))
            print("")

        plt.title('Runtime Analysis')
        plt.xlabel('Problem size')
        plt.ylabel('Runtime (s)')
        plt.errorbar(xval, naive_val, yerr=yerr1, label='Naive')
        plt.errorbar(xval, fft_val, yerr=yerr2, label='FFT')
        plt.legend(loc="upper left")
        plt.show()

    def test(self):
        print(np.allclose(dft_naive_1d(
            [1, 2, 3, 4, 5, 6]), np.fft.fft([1, 2, 3, 4, 5, 6])))
        print(np.allclose(dft_naive_2d(
            [[1, 2], [3, 4], [5, 6]]), np.fft.fft2([[1, 2], [3, 4], [5, 6]])))
        print(np.allclose(dft_inverse_1d(
            [1, 2, 3, 4, 5, 6]), np.fft.ifft([1, 2, 3, 4, 5, 6])))
        print(np.allclose(dft_inverse_2d(
            [[1, 2], [3, 4], [5, 6]]), np.fft.ifft2([[1, 2], [3, 4], [5, 6]])))
        x = np.random.rand(64)
        print(np.allclose(fft_1d(x), np.fft.fft(x)))
        print(np.allclose(fft_inverse_1d(x), np.fft.ifft(x)))
        y = np.random.rand(64, 64)
        print(np.allclose(fft_2d(y), np.fft.fft2(y)))
        print(np.allclose(fft_inverse_2d(y), np.fft.ifft2(y)))
        # self.display(abs(np.fft.fft2(self.image)))

    def image_compression(self, transformed_image, compression_level, count):
        if compression_level < 0 or compression_level > 100:
            print("Invalid compression values")
            return

        percentile = (100 - compression_level)//2
        l = np.percentile(transformed_image, percentile)
        u = np.percentile(transformed_image, 100 - percentile)
        x = np.logical_or(transformed_image <= l, transformed_image >= u)
        compressed_im_fft = transformed_image * x
        self.print_img_compression_log(compression_level, count)
        return fft_inverse_2d(compressed_im_fft)

    def print_img_compression_log(self, cl, count):
        nzv = int(count * ((100 - cl) / 100.0))
        print('non zero values for level {}% are {} out of {}'.format(cl, nzv, count))
