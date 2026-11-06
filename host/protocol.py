def decode_dot_products(packet, metadata):
    if len(packet) != 12 or metadata.get('result_semantics') != 'one_full_and_three_partial_dot_products':
        raise ValueError('Expected one complete and three partial signed 24-bit sums')
    return [int.from_bytes(packet[3*i:3*i+3], 'little', signed=True) for i in range(4)]

def encode_activations(values):
    if len(values) != 1024 or any(int(v) != v or not -128 <= v <= 127 for v in values):
        raise ValueError('Expected 1024 signed INT8 integers')
    values = [int(v) & 255 for v in values]
    return bytes(sum(((values[base + lane] >> bit) & 1) << lane for lane in range(8))
                 for bit in range(7, -1, -1) for base in range(0, 1024, 8))
