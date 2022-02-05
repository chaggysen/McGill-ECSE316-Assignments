import enum


class QueryType(enum.IntEnum):
    A = 1,
    NS = 2,
    MX = 15,
    HINFO = 13,
    SOA = 6,
    CNAME = 5,
    OTHER = 100
