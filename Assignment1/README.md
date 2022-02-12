# DNS Client
## Usage
```
usage: app.py [-h] [-t TIMEOUT] [-r MAXRETRY] [-p PORT] [-mx | -ns] IPV4ADDRESS DOMAINNAME
```

### Parameters
positional arguments:
- IPV4ADDRESS  IPv4 address
- DOMAINNAME   Domain name

optional arguments:
- -h, --help:   show this help message and exit
- -t TIMEOUT:   Timeout (sec)
- -r MAXRETRY:  Max number of retry
- -p PORT:      UDP port number
- -mx:          Send to mail server (T/F)
- -ns:          Send to name server (T/F)

## How To Run
```bash
    python app.py @8.8.8.8 www.mcgill.ca
```