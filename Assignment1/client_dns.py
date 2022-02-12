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
            print("Data sent:")
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

            print("Packet received")
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

    def refactor_response(self, rcvPacket, lenSent):

        # Unpack header
        ID, flags, QD_COUNT, AN_COUNT, NS_COUNT, AR_COUNT = self.unpack_header(
            rcvPacket)
        print("id: " + str(ID))
        print("flags: " + str(flags))
        print("qdcount: " + str(QD_COUNT))
        print("ancount: " + str(AN_COUNT))
        print("nscount: " + str(NS_COUNT))
        print("arcount: " + str(AR_COUNT))

        # Non-autoritative response
        AA = (flags & 0x0400) != 0
        # Header offset
        header_offset = 12

        questions, question_offset = self.parse_questions(
            rcvPacket, QD_COUNT, header_offset)
        print(questions)
        answers, answer_offset = self.parse_answers(
            rcvPacket, AN_COUNT, question_offset, AA)
        # authoritatives, autho_offset = self.parse_autho(
        #     rcvPacket, NS_COUNT, answer_offset, AA)
        # additionals, addi_offset = self.parse_addi(
        #     rcvPacket, AR_COUNT, autho_offset, AA)

    def parse_answers(self, packet, count, ofs, AA):
        return self.parse_sections(packet, count, ofs, AA)

    def parse_autho(self, packet, count, ofs, AA):
        return self.parse_sections(packet, count, ofs, AA)

    def parse_addi(self, packet, count, ofs, AA):
        return self.parse_sections(packet, count, ofs, AA)

    def get_name_type_class(self, packet, ofs, section):
        x_name, x_ofs = self.parse_labels(packet, ofs)
        x_type, x_class = section.unpack_from(packet, x_ofs)
        return [x_name, x_type, x_class, x_ofs]

    def format_unpack(self, struct_format, packet, ofs):
        unpack, = struct_format.unpack_from(packet, ofs)
        new_ofs = ofs + struct_format.size
        return [unpack, new_ofs]

    def build_default_record(self, authorization, a_name, a_type, a_class, ttl, rdlength):
        return {
            # change name :))))))))))))))) !!!!!!!!!!!!!!!!!!!!!!!! OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
            "auth": authorization,
            "domain_name": a_name,
            "query_type": a_type,
            "query_class": a_class,
            "ttl": ttl,
            "rdlength": rdlength
        }

    def check_type(self, x_type, record, packet, ofs):
        if x_type == self.match_query_to_int('A'):
            pass
        elif x_type == self.match_query_to_int('NS'):
            pass
        elif x_type == self.match_query_to_int("MX"):
            pass
        elif x_type == self.match_query_to_int("CNAME"):
            pass

    def parse_sections(self, packet, count, ofs, AA):
        section = struct.Struct("!2H")
        ttl_format = struct.Struct("!I")
        rd_format = struct.Struct("!H")
        records = []
        authorization = 'auth' if AA else 'noauth'
        while count != 0:
            a_name, a_type, a_class, a_ofs = self.get_name_type_class(
                packet, ofs, section)
            ofs += section.size
            ttl, ofs = self.format_unpack(ttl_format, packet, ofs)
            rdlength, ofs = self.format_unpack(rd_format, packet, ofs)
            record = self.build_default_record(
                authorization, a_name, a_type, a_class, ttl, rdlength)
            new_record, ofs = self.check_type(a_type, record, packet, ofs)
            records.append(new_record)
            count -= 1
        return records, ofs

    def parse_questions(self, packet, count, ofs):
        section = struct.Struct("!2H")
        questions = []
        for i in range(count):
            question_name, question_type, question_class, question_ofs = self.get_name_type_class(
                packet, ofs, section)
            new_question = {
                "question_name": question_name,
                "question_type": question_type,
                "question_class": question_class
            }
            questions.append(new_question)
        return questions, question_ofs + section.size

    def convert_bytes_to_qname(self, bytes):
        # not sure this actually works with number in url
        qname = ""
        for i in range(len(bytes)):
            if bytes[i:i+1].isalpha():
                qname += chr(bytes[i])
            elif i != 0:
                qname += "."
        return qname

    def unpack_header(self, rcvPacket):
        header = struct.Struct("!6H")
        return header.unpack_from(rcvPacket)

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

    # refactor this
    def parse_labels(self, packet, ofs):
        parsed_labels = []
        x = 0xC0
        while True:
            packet_len, = struct.unpack_from("!B", packet, ofs)
            if (packet_len & x) == x:
                ptr, = struct.unpack_from("!H", packet, ofs)
                ofs += 2
                return (list(parsed_labels) + list(self.parse_labels(packet, ptr & 0x3FFF))), ofs
            if (packet_len & x) != 0:
                raise StandardError("unknown label encoding")
            ofs += 1
            if packet_len == 0:
                return parsed_labels, ofs
            parsed_labels.append(*struct.unpack_from("!%ds" %
                                                     packet_len, packet, ofs))
            ofs += packet_len
