"""Fallback: stream-inflate train.csv from a truncated UCI Porto zip (nested stored train.csv.zip -> deflated train.csv).
Writes complete lines only. Usage: python salvage_porto.py <truncated zip> <out csv>"""
import sys, struct, zlib
src, dst = sys.argv[1], sys.argv[2]
f = open(src, 'rb'); data = f.read()
o = data.find(b'PK\x03\x04train.csv'[:4], 451400)
o = 451456
sig, ver, flag, meth, mt, md, crc, cs, us, nl, el = struct.unpack('<IHHHHHIIIHH', data[o:o + 30])
assert data[o + 30:o + 30 + nl] == b'train.csv' and meth == 8
start = o + 30 + nl + el
d = zlib.decompressobj(-15); out = open(dst, 'wb'); buf = b''; nbytes = 0
for i in range(start, len(data), 1 << 22):
    try: chunk = d.decompress(data[i:i + (1 << 22)])
    except zlib.error as e: print('zlib error', e); break
    buf += chunk; cut = buf.rfind(b'\n')
    if cut >= 0: out.write(buf[:cut + 1]); nbytes += cut + 1; buf = buf[cut + 1:]
out.close(); print('wrote', nbytes, 'bytes')
