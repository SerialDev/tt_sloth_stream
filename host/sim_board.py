"""Public-pin SDK adapter; no direct access to chip arithmetic state."""
import ctypes, subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1]; b=r/"build/bridge"
b.mkdir(parents=True,exist_ok=True)
# Incremental rebuild prevents accidentally testing a stale RTL library.
subprocess.run(['verilator','--cc','--build','-j','4','--top-module','tt_um_serialdev_sloth_stream','--Mdir',str(b),'-CFLAGS','-fPIC -O3','-LDFLAGS','-shared','--exe',str(r/'test/bridge.cpp'),str(r/'src/project.v'),'-o','libstream.so'],check=True,stdout=subprocess.DEVNULL)
lib=ctypes.CDLL(str(b/'libstream.so'));lib.stream_create.restype=ctypes.c_void_p
for name in ['stream_destroy','stream_tick','stream_flags','stream_data','stream_oe']:
 getattr(lib,name).argtypes=[ctypes.c_void_p]
lib.stream_pins.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
class Pin:
 def __init__(self,value=0):self.value=value
class Read:
 def __init__(self,fn,p):self.fn=fn;self.p=p
 @property
 def value(self):return self.fn(self.p)
class TT:
 def __init__(self):
  self.p=lib.stream_create();self.reset=1;self.cycles=0
  self.ui_in=Pin();self.uio_in=Pin();self.uio_oe_pico=Pin()
  self.uio_out=Read(lib.stream_flags,self.p);self.uo_out=Read(lib.stream_data,self.p)
 def clock_project_stop(self):pass
 def clk(self,value):assert value==0
 def reset_project(self,active):
  self.reset=not active;lib.stream_pins(self.p,self.ui_in.value,self.uio_in.value,self.reset)
 def clock_project_once(self,msDelay=0):
  assert self.uio_oe_pico.value==7, 'RP must drive only control bits 0-2'
  assert not (lib.stream_oe(self.p)&self.uio_oe_pico.value), 'Board/chip output contention'
  lib.stream_pins(self.p,self.ui_in.value,self.uio_in.value,self.reset);lib.stream_tick(self.p);self.cycles+=1

 def close(self):
  lib.stream_destroy(self.p)
