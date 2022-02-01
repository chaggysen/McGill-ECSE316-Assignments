import socket
import time

class DNS_CLIENT:
    def __init__(self, args):
        self.parseCommandArguments(args)
        self.send_query(1)

    def parseCommandArguments(self, args):
        self.timeout = args.TIMEOUT
        self.MAXRETRIES = args.MAXRETRY
        self.port = args.PORT
        if args.MAILSERVER:
            self.qType = "MX"
        elif args.NAMESERVER:
            self.qType = "NS"
        else:
            self.qType = "A"
        self.address = args.IPV4ADDRESS.replace("@", "")
        self.name = args.DOMAINNAME
        self.server = self.address + "." + self.name

    def send_query(self, retries):
        # make request
        if retries == 1:
            requestDescription = "Dns Client sending request for " + self.name + "\n" + \
                "Server: " + self.address + "\n" + \
                "Request type: " + self.qType + "\n"
            print(requestDescription)
        pass

        if retries > self.MAXRETRIES:
            print("Maximum retries reached. Exiting...")
            return

        try:
            # Setting up socket
            # DGRAM instead of STREAM
            clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            clientSocket.settimeout(self.timeout)

            # Create InetAddress used by datagram packet
            ipAddress = socket.inet_aton(self.address)

            # Setting up request TODO
            sendData = self.constructRequest()
            receiveData = bytearray(1024)

            # Setting up packets
            clientPacket = (ipAddress, self.port)
            receivePacket = (receiveData, 1024)

            # measure the time taken to send and receive query
            start = time.time()
            clientSocket.sendto(sendData, clientPacket)
            clientSocket.recvfrom(receivePacket)
            end = time.time()
            clientSocket.close()

            print("Response received after " + (end - start) + " seconds " + "(" + (retries - 1) + " retries)")

            # TODO
            res = self.refactor_response()
            self.print_response(res)

        except socket.timeout as msg:
            print("Timeout reached. Retrying...")
            self.send_query(retries + 1)
        except socket.error as msg:
            print("Error: " + str(msg))
            return


    def refactor_response(self):
        pass

    def constructRequest(self):
        pass

    def print_response(self, response):
        pass