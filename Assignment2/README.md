# FFT Visualization program
## Usage
```
usage: fft.py [-h] [-m MODE] [-i IMAGE]
```

### Parameters
optional arguments:
-  -h, --help:  show this help message and exit
-  -m MODE (default 1):     The mode to select (1 for fft, 2 for denoising, 3 for compression, 4 for runtime analysis)
-  -i IMAGE (default moonlanding.jpg):    Path to the image file to be processed


## How To Run
```bash
    python fft.py -m 1 -i moonlanding.png
```