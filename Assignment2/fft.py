import argparse
from fourier_transform import FOURIER_TRANSFORM


def __main__():
    args = set_arguments()
    FOURIER_TRANSFORM(args)

def set_arguments():
    parser = argparse.ArgumentParser(description="Fourier Transform!")
    parser.add_argument("-m", type=int, help="Mode",
                        dest="MODE", default=1)
    parser.add_argument(
        "-i", help="Image", dest="IMAGE", default='moonlanding.png')

    return parser.parse_args()


if __name__ == "__main__":
    __main__()
