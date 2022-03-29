import numpy as np


def dft_naive_1d_explicit(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    X = np.zeros(N, dtype=complex)
    # sum from n=0 to N-1 of x[n]*exp(-i*2*pi/N*k*n) where k=0,1,...,N-1
    for n in range(N):
        for k in range(N):
            X[k] += x[n]*np.exp(-2j*np.pi/N*k*n)
    return X

def dft_inverse_1d_explicit(X):
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    x = np.zeros(N, dtype=complex)
    # sum from n=0 to N-1 of 1/N*X[k]*exp(i*2*pi/N*k*n) where k=0,1,...,N-1
    for n in range(N):
        for k in range(N):
            x[n] += 1/N*X[k]*np.exp(2j*np.pi/N*k*n)
    return x

def dft_naive_2d_explicit(f):
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

def dft_inverse_2d_explicit(F):
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

def fft_1d_explicit(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    X = np.zeros(N, dtype=complex)
    if N % 2 != 0:
        raise ValueError("size of x must be a power of 2")
    # if the array is small enough, use the naive method
    if N <= 16:
        X = dft_naive_1d_explicit(x)
    else:
        # split the array into two halves
        X_even = fft_1d_explicit(x[::2])
        X_odd = fft_1d_explicit(x[1::2])
        # compute the sum of the two halves
        half = N//2
        for k in range(half):
            p = X_even[k]
            q = np.exp(-2j*np.pi/N*k)*X_odd[k]
            X[k] = p + q
            X[k+half] = p - q
    return X

def fft_inverse_1d_explicit(X):
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    x = np.zeros(N, dtype=complex)
    if N % 2 != 0:
        raise ValueError("size of X must be a power of 2")
    # if the array is small enough, use the naive method
    if N <= 16:
        x = dft_inverse_1d_explicit(X)
    else:
        # split the array into two halves
        X_even = fft_inverse_1d_explicit(X[::2])
        X_odd = fft_inverse_1d_explicit(X[1::2])
        # compute the sum of the two halves
        half = N//2
        for k in range(half):
            p = 1/2*X_even[k]
            q = 1/2*np.exp(2j*np.pi/N*k)*X_odd[k]
            x[k] = p + q
            x[k+half] = p - q
    return x

# --- Methods below this line are significantly more efficient than the methods above ---

def dft_naive_1d(x):
    # Computes DFT of 1D array x
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j*np.pi/N*k*n)
    return np.dot(M, x)

def dft_inverse_1d(X):
    # Computes IDFT of 1D array X
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = 1/N*np.exp(2j*np.pi/N*k*n)
    return np.dot(M, X)

def dft_naive_2d(f):
    f = np.asarray(f, dtype=complex)
    M = f.shape[0]
    N = f.shape[1]
    F = np.zeros((M, N), dtype=complex)
    # use dft_naive_1d to compute the DFT of each column
    for n in range(N):
        F[:,n] = dft_naive_1d(f[:,n])
    # use dft_naive_1d to compute the DFT of each row
    for m in range(M):
        F[m,:] = dft_naive_1d(F[m,:])
    return F

def dft_inverse_2d(F):
    F = np.asarray(F, dtype=complex)
    M = F.shape[0]
    N = F.shape[1]
    f = np.zeros((M, N), dtype=complex)
    # use dft_inverse_1d to compute the IDFT of each column
    for n in range(N):
        f[:,n] = dft_inverse_1d(F[:,n])
    # use dft_inverse_1d to compute the IDFT of each row
    for m in range(M):
        f[m,:] = dft_inverse_1d(f[m,:])
    return f

def fft_1d(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    X = np.zeros(N, dtype=complex)
    if N % 2 != 0:
        raise ValueError("size of x must be a power of 2")
    # if the array is small enough, use the naive method
    if N <= 16:
        X = dft_naive_1d(x)
    else:
        # split the array into two halves
        X_even = fft_1d(x[::2])
        X_odd = fft_1d(x[1::2])
        # compute the sum of the two halves
        half = N//2
        factor = np.exp(-2j*np.pi*np.arange(N)/N)
        X = np.concatenate([X_even + factor[:half]*X_odd, X_even + factor[half:]*X_odd])
    return X

def fft_inverse_1d(X):
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    x = np.zeros(N, dtype=complex)
    if N % 2 != 0:
        raise ValueError("size of X must be a power of 2")
    # if the array is small enough, use the naive method
    if N <= 16:
        x = dft_inverse_1d(X)
    else:
        # split the array into two halves
        X_even = fft_inverse_1d(X[::2])
        X_odd = fft_inverse_1d(X[1::2])
        # compute the sum of the two halves
        half = N//2
        factor = np.exp(2j*np.pi*np.arange(N)/N)
        x = 1/2*np.concatenate([X_even + factor[:half]*X_odd, X_even + factor[half:]*X_odd])
    return x

def fft_2d(f):
    f = np.asarray(f, dtype=complex)
    M = f.shape[0]
    N = f.shape[1]
    F = np.zeros((M, N), dtype=complex)
    
    # use fft_1d to compute the FFT of each column
    for n in range(N):
        F[:,n] = fft_1d(f[:,n])
    # use fft_1d to compute the FFT of each row
    for m in range(M):
        F[m,:] = fft_1d(F[m,:])
    return F

def fft_inverse_2d(F):
    F = np.asarray(F, dtype=complex)
    M = F.shape[0]
    N = F.shape[1]
    f = np.zeros((M, N), dtype=complex)
    
    # use fft_inverse_1d to compute the inverse FFT of each column
    for n in range(N):
        f[:,n] = fft_inverse_1d(F[:,n])
    # use fft_inverse_1d to compute the inverse FFT of each row
    for m in range(M):
        f[m,:] = fft_inverse_1d(f[m,:])
    return f