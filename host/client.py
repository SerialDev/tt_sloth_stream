from protocol import encode_activations
"""The same host client runs against simulated pins or board USB serial."""
from wire import frame, parse, read_line, write_all
from device import IDENTITY


class Client:
    def __init__(self, stream):
        self.stream = stream
        if self.exchange(b'I', b'', b'I') != IDENTITY:
            raise RuntimeError('Firmware identity mismatch')

    def exchange(self, kind, payload, expected):
        write_all(self.stream, frame(kind, payload))
        line = read_line(self.stream)
        if line is None:
            raise TimeoutError('No response; reconnect and reset')
        kind, payload = parse(line)
        if kind != expected:
            raise RuntimeError('Device rejected request or response out of sequence')
        return payload

    def dot_products(self, values):
        if len(values) != 1024 or any(int(v) != v or not -128 <= v <= 127 for v in values):
            raise ValueError('Expected 1024 signed INT8 integers')
        payload = encode_activations(values)
        result = self.exchange(b'W', payload, b'R')
        if len(result) != 12:
            raise RuntimeError('Invalid result length')
        return result
