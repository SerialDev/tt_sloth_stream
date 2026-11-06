"""MicroPython bring-up driver for the selected streaming shard.

Manual clocking is for validation, not a line-rate or USB performance claim.
The selected project must match the metadata used by the host decoder.
"""
class Board:
    def __init__(self,tt):
        self.tt=tt
        tt.clock_project_stop();tt.clk(0)
        tt.uio_oe_pico.value=7
        tt.uio_in.value=0;tt.ui_in.value=0
        tt.reset_project(True);self.tick();tt.reset_project(False);self.tick()
    def tick(self,data=0,ctrl=0):
        self.tt.ui_in.value=data&255;self.tt.uio_in.value=ctrl&7
        self.tt.clock_project_once(msDelay=0)
    def abort(self):
        self.tick(ctrl=4)
    def dot_products(self,activations):
        if len(activations)!=1024:
            raise ValueError('Expected 1024 signed INT8 activations')
        for value in activations:
            if int(value)!=value or not -128<=int(value)<=127:
                raise ValueError('Activation out of INT8 range')
        for value in activations:
            for _ in range(40):
                if self.tt.uio_out.value&8:break
                self.tick()
            else:
                self.abort();raise RuntimeError('Input handshake timeout')
            self.tick(int(value),1)
        for _ in range(1120):
            if self.tt.uio_out.value&16:break
            self.tick()
        else:
            self.abort();raise RuntimeError('Compute timeout')
        packet=bytearray()
        for _ in range(12):
            if not self.tt.uio_out.value&16:
                self.abort();raise RuntimeError('Truncated result')
            packet.append(self.tt.uo_out.value);self.tick(ctrl=2)
        return packet
