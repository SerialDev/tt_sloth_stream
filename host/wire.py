"""ASCII frames keep MicroPython USB REPL control bytes out of payloads.
CRC detects transmission damage, not malicious tampering. No retries: reset
and reconnect after an ambiguous timeout. Compatible with MicroPython.
"""
try:
    import ubinascii as binascii
except ImportError:
    import binascii


def crc16(data):
    crc = 0xffff
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xffff
    return crc


def frame(kind, payload=b''):
    body = kind + b':' + binascii.hexlify(payload)
    return body + b':' + ('%04x' % crc16(body)).encode() + b'\n'


def parse(line):
    parts = line.rstrip(b'\r\n').split(b':')
    if len(parts) != 3 or len(parts[0]) != 1:
        raise ValueError('Malformed frame')
    body = parts[0] + b':' + parts[1]
    if len(parts[2]) != 4 or int(parts[2], 16) != crc16(body):
        raise ValueError('CRC mismatch')
    return parts[0], binascii.unhexlify(parts[1])


def read_line(stream, limit=2112):
    data = bytearray()
    overflow = False
    while True:
        part = stream.read(1)
        if not part:
            if not data and not overflow:
                return None
            raise TimeoutError('Incomplete frame; reconnect and reset')
        if part == b'\n':
            if overflow:
                raise ValueError('Frame too long')
            return bytes(data) + part
        if len(data) < limit:
            data.extend(part)
        else:
            overflow = True


def write_all(stream, data):
    while data:
        count = stream.write(data)
        if count is None or count <= 0:
            raise OSError('Transport write failed')
        data = data[count:]
    if hasattr(stream, 'flush'):
        stream.flush()
