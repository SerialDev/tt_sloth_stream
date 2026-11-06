"""Transport failure handling, independent of arithmetic known-answer tests."""
import io
from client import Client
from device import IDENTITY
from wire import frame,parse,read_line,crc16
assert crc16(b'123456789')==0x29b1
assert parse(frame(b'W',bytes(range(256))))==(b'W',bytes(range(256)))
class Scripted:
 def __init__(self,response):self.input=io.BytesIO(response);self.written=bytearray()
 def read(self,n):return self.input.read(n)
 def write(self,data):self.written.extend(data[:3]);return min(3,len(data))

cases=0
for reply,exception in [(b'',TimeoutError),(b'R:00',TimeoutError),(b'R:00:0000\n',ValueError),(frame(b'R',b'\0'),RuntimeError),(frame(b'E',b'rejected'),RuntimeError)]:
 stream=Scripted(frame(b'I',IDENTITY)+reply);client=Client(stream)
 try:client.dot_products([0]*1024)
 except exception:cases+=1
 else:raise AssertionError('Invalid response accepted')
 assert stream.written.count(b'W:')==1,'Must not retry ambiguous request'
stream=Scripted(frame(b'I',IDENTITY));client=Client(stream);before=bytes(stream.written)
for bad in [[],[128]*1024,[0.5]*1024,[-129]*1024]:
 try:client.dot_products(bad)
 except ValueError:cases+=1
 else:raise AssertionError('Invalid input accepted')
assert bytes(stream.written)==before
try:Client(Scripted(frame(b'I',b'wrong firmware')))
except RuntimeError:cases+=1
else:raise AssertionError('Wrong firmware accepted')
print('PASS',cases,'transport/validation failures; no ambiguous retries; CRC known answer')
