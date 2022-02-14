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
            print("ERROR\t Maximum number of retries (" +
                  str(self.MAXRETRIES) + ") exceeded")
            return

        try:
            # Setting up socket
            clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            clientSocket.settimeout(self.timeout)

            sendData = self.construct_request(self.name, self.qType)
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

            print("Response received after " + str(end - start) +
                  " seconds " + "(" + str(retries - 1) + " retries)")

            res = self.refactor_response(receivePacket)
            self.print_response(res)

        except socket.timeout as msg:
            print("Timeout reached. Retrying...")
            self.send_query(retries + 1)
        except socket.error as msg:
            print("ERROR\t" + str(msg))
            return

    def print_response(self, response):
        ancount = response['an_count']
        arcount = response['ar_count']

        if ancount + arcount <= 0:
            print("NOT FOUND")
            return

        if ancount > 0:
            print("***Answer Section (" + str(ancount) + " records)***")
            for answer in response['answers']:
                print(answer)

        print()

        if arcount > 0:
            print("***Additional Section (" + str(arcount) + " records)***")
            for add in response['additionals']:
                print(add)

    # ------------- Request Section -------------

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

    def construct_request(self, name, q_type):
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

    # ------------- Response Section -------------

    def refactor_response(self, rcvPacket):

        # Unpack header
        ID, flags, QD_COUNT, AN_COUNT, NS_COUNT, AR_COUNT = self.unpack_header(
            rcvPacket)

        # Header offset
        header_offset = 12

        # Non-autoritative response
        AA = (flags & 0x0400) != 0

        questions, question_offset = self.parse_questions(
            rcvPacket, QD_COUNT, header_offset)
        answers, answer_offset = self.parse_answers(
            rcvPacket, AN_COUNT, question_offset, AA)
        authoritatives, autho_offset = self.parse_autho(
            rcvPacket, NS_COUNT, answer_offset, AA)
        additionals, addi_offset = self.parse_addi(
            rcvPacket, AR_COUNT, autho_offset, AA)
        output = {
            'id': ID,
            'aa': AA,
            'qd_count': QD_COUNT,
            'an_count': AN_COUNT,
            'ns_count': NS_COUNT,
            'ar_count': AR_COUNT,
            'questions': questions,
            'answers': answers,
            'additionals': additionals
        }
        return output

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

    def get_ip(self, packet, ofs):
        v1, v2, v3, v4 = struct.unpack_from("!4B", packet, ofs)
        ip_value = "" + str(v1) + '.' + str(v2) + '.' + str(v3) + '.' + str(v4)
        return ip_value

    def update_record(self, record, attribute, value):
        record[attribute] = value
        return record

    def check_type(self, x_type, record, packet, ofs):
        if x_type == self.match_query_to_int('A'):
            IP_OFFSET = 4
            ip_value = self.get_ip(packet, ofs)
            new_record = self.update_record(record, "address", ip_value)
            ofs += IP_OFFSET
            return [self.format_a_record(
                    new_record['address'], new_record['ttl'], new_record['auth']), ofs]
        elif x_type == self.match_query_to_int('NS'):
            ns, new_ofs = self.parse_labels(packet, ofs)
            new_record = self.update_record(record, 'name_server', ns)
            return [self.format_ns_record(new_record['name_server'], new_record['ttl'], new_record['auth']), new_ofs]
        elif x_type == self.match_query_to_int("MX"):
            pref_format = struct.Struct("!H")
            pref_val, ofs = self.format_unpack(pref_format, packet, ofs)
            exch_val, ofs = self.parse_labels(packet, ofs)
            record = self.update_record(record, "preference", pref_val)
            record = self.update_record(record, "exchange", exch_val)
            return [self.format_mx_record(record['exchange'], record['preference'], record['ttl'], record['auth']), ofs]
        elif x_type == self.match_query_to_int("CNAME"):
            c_name_val, ofs = self.parse_labels(packet, ofs)
            record = self.update_record(record, "cname", c_name_val)
            return [self.format_cname_record(record['cname'], record['ttl'], record['auth']), ofs]
        else:
            rdlength = record['rdlength']
            rdata = packet[ofs:ofs+rdlength]
            ofs += rdlength
            new_record = self.update_record(record, "rdata", rdata)
            return [self.format_other_record(new_record['d_name'], new_record['ttl'], new_record['auth']), ofs]

    def build_default_record(self, authorization, a_name, a_type, a_class, ttl, rdlength):
        return {
            "auth": authorization,
            "d_name": a_name,
            "query_type": a_type,
            "query_class": a_class,
            "ttl": ttl,
            "rdlength": rdlength
        }

    def parse_sections(self, packet, count, ofs, AA):
        section = struct.Struct("!2H")
        ttl_format = struct.Struct("!I")
        rd_format = struct.Struct("!H")
        records = []
        authorization = 'auth' if AA else 'noauth'
        while count != 0:
            a_name, a_type, a_class, ofs = self.get_name_type_class(
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

        
    def unpack_header(self, rcvPacket):
        header = struct.Struct("!6H")
        return header.unpack_from(rcvPacket)

    # refactor this TODO
    def parse_labels(self, packet, ofs):
        parsed_labels = []
        x = 0xC0
        while True:
            packet_len, = struct.unpack_from("!B", packet, ofs)
            if (packet_len & x) == x:
                ptr, = struct.unpack_from("!H", packet, ofs)
                ofs += 2
                return (list(parsed_labels) + list(self.parse_labels(packet, ptr & 0x3FFF))), ofs
            ofs += 1
            if packet_len == 0:
                return parsed_labels, ofs
            parsed_labels.append(*struct.unpack_from("!%ds" %
                                                     packet_len, packet, ofs))
            ofs += packet_len

    # ----------------- Response Formatting -----------------

    def flatten(self, lst):
        if lst == []:
            return lst
        if isinstance(lst[0], list):
            return self.flatten(lst[0]) + self.flatten(lst[1:])
        return lst[:1] + self.flatten(lst[1:])

    def format_a_record(self, address, ttl, auth):
        return "IP\t" + address + "\t" + str(ttl) + "\t" + auth

    def format_cname_record(self, cname, ttl, auth):
        flattened = self.flatten(cname)
        flattened = [x for x in flattened if not isinstance(x, int)]
        flattened = b'.'.join(flattened).decode('utf-8')
        return "CNAME\t" + flattened + "\t" + str(ttl) + "\t" + auth

    def format_mx_record(self, exchange, preference, ttl, auth):
        flattened = self.flatten(exchange)
        flattened = [x for x in flattened if not isinstance(x, int)]
        flattened = b'.'.join(flattened).decode('utf-8')
        return "MX\t" + flattened + "\t" + str(preference) + "\t" + str(ttl) + "\t" + auth

    def format_ns_record(self, name_server, ttl, auth):
        flattened = self.flatten(name_server)
        flattened = [x for x in flattened if not isinstance(x, int)]
        flattened = b'.'.join(flattened).decode('utf-8')
        return "NS\t" + flattened + "\t" + str(ttl) + "\t" + auth

    def format_other_record(self, name, ttl, auth):
        flattened = self.flatten(name)
        flattened = [x for x in flattened if not isinstance(x, int)]
        flattened = b'.'.join(flattened).decode('utf-8')
        return "OTHER\t" + flattened + "\t" + str(ttl) + "\t" + auth