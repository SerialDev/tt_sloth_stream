"""Board USB service. Select the project before calling serve(Board(tt))."""
from wire import frame, parse, read_line, write_all
# Firmware identity only: verify silicon identity with known-answer vectors.
IDENTITY = b'SLOTH1:6ed3d5cc2ddcb5ed9a8a2eb38cfaab8fe0215d550ef2154543a6333cddada551'


def serve(board, reader=None, writer=None):
    if reader is None:
        import sys
        reader, writer = sys.stdin.buffer, sys.stdout.buffer
    while True:
        try:
            line = read_line(reader)
            if line is None:
                return
            command, payload = parse(line)
            if command == b'I' and not payload:
                response = frame(b'I', IDENTITY)
            elif command == b'W' and len(payload) == 1024:
                values = [v if v < 128 else v - 256 for v in payload]
                response = frame(b'R', board.dot_products(values))
            elif command == b'A' and not payload:
                board.abort()
                response = frame(b'K')
            else:
                raise ValueError('Invalid command or payload length')
        except (ValueError, RuntimeError):
            board.abort()
            response = frame(b'E', b'rejected')
        write_all(writer, response)
