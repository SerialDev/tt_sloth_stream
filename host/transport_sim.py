"""Socket byte stream and real device service backed by Verilated public pins."""
import socket
import threading
from sim_board import TT
from board import Board
from device import serve


class Stream:
    def __init__(self, sock):
        self.sock = sock
        self.sent = 0
    def read(self, n):
        return self.sock.recv(min(n, 7))
    def write(self, data):
        n = self.sock.send(data[:17])  # force partial writes
        self.sent += n
        return n


class Simulation:
    def __init__(self):
        self.host_sock, self.device_sock = socket.socketpair()
        self.host_sock.settimeout(10)
        self.stream = Stream(self.host_sock)
        self.device_stream = Stream(self.device_sock)
        self.tt = TT()
        self.board = Board(self.tt)
        self.errors = []
        def run():
            try:
                serve(self.board, self.device_stream, self.device_stream)
            except Exception as error:
                self.errors.append(repr(error))
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
    def close(self):
        self.host_sock.shutdown(socket.SHUT_WR)
        self.thread.join(5)
        self.host_sock.close()
        self.device_sock.close()
        if self.thread.is_alive():
            raise RuntimeError('Device service did not stop')
        self.tt.close()
        if self.errors:
            raise RuntimeError(self.errors)
