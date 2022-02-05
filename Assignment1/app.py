import argparse
from client_dns import DNS_CLIENT


def __main__():
    print("Hello World")
    args = set_arguments()
    print(args)
    DNS_CLIENT(args)


def set_arguments():
    parser = argparse.ArgumentParser(description="DNS Client!")
    parser.add_argument("-t", type=int, help="Timeout (sec)",
                        dest="TIMEOUT", default=5)
    parser.add_argument(
        "-r", type=int, help="Max number of retry", dest="MAXRETRY", default=3)
    parser.add_argument(
        "-p", type=int, help="UDP port number", dest="PORT", default=53)
    mx_group = parser.add_mutually_exclusive_group(required=False)
    mx_group.add_argument(
        "-mx", help="Send to mail server (T/F)", dest="MAILSERVER", default=False, action="store_true")
    mx_group.add_argument(
        "-ns", help="Send to name server (T/F)", dest="NAMESERVER", default=False, action="store_true")
    parser.add_argument(help="IPv4 address", dest="IPV4ADDRESS")
    parser.add_argument(help="Domain name", dest="DOMAINNAME")

    return parser.parse_args()


if __name__ == "__main__":
    __main__()
