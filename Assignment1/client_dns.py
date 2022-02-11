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
        # NAME, TYPE, CLASS, TTL, RDLENGTH, RDATA(PREFERENCE, EXCHANGE)
        # get only the answer part
        answer = rcvPacket[lenSent:]
        # get the name
        name = ""
        first = struct.unpack('B', answer[0:1])[0]
        if first == 192: #pointer (need to better integrate this for the rest)
            offset = struct.unpack('>H', answer[0:2])[0] - 49152 # c000 is 49152
            nameStart = rcvPacket[offset:]
            label = nameStart[0]
            end = 0
            while label != 0:
                print(label)
                end += label + 1
                label = nameStart[end]
            name = nameStart[:end]
        else: # not pointer
            nameStart = answer[1:]
            label = nameStart[0]
            end = 0
            while label != 0:
                end += label + 1
                label = nameStart[end]
            name = nameStart[:end]
            answer = answer[end:] 
        # get the actual name
        actualName = self.convert_bytes_to_qname(name)
        print("Name: " + str(actualName))

        # get the TYPE
        type = struct.unpack('>H', answer[2:4])[0]
        type = self.match_type_to_query(type)
        print("Type: " + type)

        # get the CLASS
        class_ = struct.unpack('>H', answer[4:6])[0]
        print("Class: " + str(class_))
        if class_ != 1:
            print("Class is not 1. This is not a valid DNS response.")
            return

        # get the TTL
        ttl = struct.unpack('>I', answer[6:10])[0]
        print("TTL: " + str(ttl))

        # get the RDLENGTH
        rdlength = struct.unpack('>H', answer[10:12])[0]
        print("RDLENGTH: " + str(rdlength))

        # get the RDATA
        rdata = answer[12:12 + rdlength]
        if type == 'A':
            ipAddr = str(rdata[0]) + '.' + str(rdata[1]) + '.' + str(rdata[2]) + '.' + str(rdata[3])
            print("IP Address: " + str(ipAddr))
        elif type == 'NS':
            name = rdata[0]
            # get the actual name
            actualName = self.convert_bytes_to_qname(name)
            print("Server Name: " + str(actualName))
        elif type == 'CNAME': # not sure for this part
            name = rdata[0]
            # get the actual name
            actualName = self.convert_bytes_to_qname(name)
            print("Alias Name: " + str(actualName))
        elif type == 'MX':
            preference = struct.unpack('>H', rdata[0:2])[0]
            name = rdata[2]
            # get the actual name
            actualName = self.convert_bytes_to_qname(name)
            print("Preference: " + str(preference))
            print("Exchange Name: " + str(actualName))

    def convert_bytes_to_qname(self, bytes):
        # not sure this actually works with number in url
        qname = ""
        for i in range(len(bytes)):
            if bytes[i:i+1].isalpha():
                qname += chr(bytes[i])
            elif i != 0:
                qname += "."
        return qname

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

    def match_type_to_query(self, type):
        type_dict = {
            1: 'A',
            2: 'NS',
            5: 'CNAME',
            6: 'SOA',
            13: 'HINFO',
            15: 'MX',
            100: 'OTHER'
        }
        if type in type_dict:
            return type_dict[type]
        else:
            return 'A'

    def print_response(self, response):
        ancount = response['ancount']
        arcount = response['arcount']

        if ancount + arcount <= 0:
            print("NOT FOUND")
            return

        if ancount > 0:
            print("***Answer Section (" + ancount + " records)***")
            for answer in response['answers']:
                print(str(answer))

        print()

        if arcount > 0:
            print("***Additional Section (" + arcount + " records)***")
            for add in response['additional']:
                print(str(add))