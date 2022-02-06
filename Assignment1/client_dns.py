import socket
import time
import struct
import random


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

        if retries > self.MAXRETRIES:
            print("Maximum retries reached. Exiting...")
            return

        try:
            # Setting up socket
            clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            clientSocket.settimeout(self.timeout)

            sendData = self.constructRequest(self.name, self.qType)
            print(sendData)
            receiveData = bytearray(1024)

            # Setting up packets
            clientPacket = (self.address, self.port)
            receivePacket = (receiveData, 1024)

            # measure the time taken to send and receive query
            start = time.time()
            clientSocket.sendto(sendData, clientPacket)
            receivePacket, receiveaAddress = clientSocket.recvfrom(1024)
            end = time.time()
            clientSocket.close()

            print(receivePacket)
            print(receiveaAddress)

            print("Response received after " + str(end - start) +
                  " seconds " + "(" + str(retries - 1) + " retries)")

            res = self.refactor_response(receivePacket, len(sendData))
            self.print_response(res)

        except socket.timeout as msg:
            print("Timeout reached. Retrying...")
            self.send_query(retries + 1)
        except socket.error as msg:
            print("Error: " + str(msg))
            return

    def refactor_response(self, rcvPacket, lenSent):
        # id, flags, QD_Count, AN_Count, NS_Count, AR_Count
        # NAME, TYPE, CLASS, TTL, RDLENGTH, RDATA, PREFERENCE, EXCHANGE
        pass

    def construct_header(self):
        header = bytes()
        # id, flags, QD_Count, AN_Count, NS_Count, AR_Count
        items = [random.getrandbits(16), 256, 1, 0, 0, 0]
        for item in items:
            header += struct.pack(">H", item)
        return header

    def construct_question(self, name):
        question = bytes()
        url_parts = name.split('.')
        for label in url_parts:
            # a label with n char follows
            question += struct.pack('B', len(label))
            for char in label:
                question += struct.pack('c', char.encode('utf-8'))
        return question

    def construct_footer(self, q_type):
        QNAME_end, Q_Class = 0, 1
        footer = bytes()
        footer += struct.pack('B', QNAME_end)
        footer += struct.pack('>H', self.match_query_to_int(q_type))
        footer += struct.pack(">H", Q_Class)
        return footer

    def constructRequest(self, name, q_type):
        request = bytes()
        header = self.construct_header()
        question = self.construct_question(name)
        footer = self.construct_footer(q_type)
        request += (header + question + footer)
        return request

    def match_query_to_int(self, q_type):
        q_type_dict = {
            'A': 1,
            'NS': 2,
            'CNAME': 5,
            'SOA': 6,
            'HINFO': 13,
            'MX': 15,
            "OTHER": 100
        }
        if q_type in q_type_dict:
            return q_type_dict[q_type]
        else:
            return q_type_dict['A']

    def print_response(self, response):
        pass
