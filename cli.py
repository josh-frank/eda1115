import time
from smbus2 import SMBus   # `import smbus` works too: SMBus(1) is the same API

ADDR = 0x48                # ADS1115 default (ADDR pin to GND)
REG_CONV, REG_CONFIG = 0x00, 0x01

# Config: AIN0 vs GND, ±4.096 V (same as Adafruit gain=1), continuous, 128 SPS, comparator off
CONFIG = 0x4000 | 0x0200 | 0x0000 | 0x0080 | 0x0003
LSB = 4.096 / 32768        # volts per count at this gain

R_REF, VCC = 100_000, 3.3

bus = SMBus(1)
bus.write_i2c_block_data(ADDR, REG_CONFIG, [CONFIG >> 8, CONFIG & 0xFF])
time.sleep(0.02)

def read_voltage():
    hi, lo = bus.read_i2c_block_data(ADDR, REG_CONV, 2)   # ADS1115 is big-endian
    return int.from_bytes([hi, lo], "big", signed=True) * LSB

avg = None
try:
    while True:
        v = read_voltage()
        if 0.001 < v < VCC - 0.001:
            g = (VCC - v) / (v * R_REF) * 1e6                  # microsiemens
            avg = g if avg is None else 0.8 * avg + 0.2 * g    # light smoothing
            bar = "#" * int(min(avg, 30) * 2)                  # 0-30 uS -> 0-60 chars
            line = f"{v:5.3f} V  {g:6.2f} uS  avg {avg:6.2f}  {bar}"
        else:
            line = f"{v:5.3f} V  out of range, check contact"
        print("\r" + line.ljust(100), end="", flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print()
finally:
    bus.close()