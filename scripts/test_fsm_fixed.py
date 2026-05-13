import sys

payload = [0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x11, 0x22, 0x33, 0x44] # offset_L, offset_H, size_L, size_H, data0, data1, data2

buf_raddr = 0
buf_rdata = 0

state = "DISPATCH"
img_idx = 0
img_offset = 0
img_size = 0

def posedge():
    global buf_rdata
    # In Verilog: buf_rdata <= payload_buf[buf_raddr]
    # This means buf_rdata gets the value at the OLD buf_raddr
    buf_rdata = payload[buf_raddr] if buf_raddr < len(payload) else 0xFF

# Cycle 0: DISPATCH -> IMG_INIT
buf_raddr = 0
state = "IMG_INIT"
posedge()

# Cycle 1: IMG_INIT
buf_raddr = 1
img_idx = 0
state = "IMG_READ_HDR"
posedge()

# Cycle 2..5: IMG_READ_HDR
for i in range(4):
    # These assignments happen sequentially in python, but in Verilog they are parallel using OLD values.
    # So we use the current buf_rdata.
    if img_idx == 0:
        img_offset_l = buf_rdata
        buf_raddr = 2
    elif img_idx == 1:
        img_offset_h = buf_rdata
        buf_raddr = 3
    elif img_idx == 2:
        img_size_l = buf_rdata
        buf_raddr = 4
    elif img_idx == 3:
        img_size_h = buf_rdata
        img_idx = 0
        buf_raddr = 5  # <--- FIX: changed 4 to 5
        state = "IMG_WRITE"
    if state == "IMG_READ_HDR":
        img_idx += 1
    posedge()

print(f"Header done. offset={(img_size_h<<8)|img_size_l}, size={(img_size_h<<8)|img_size_l}")

# Cycle 6..9: IMG_WRITE
img_size = 4
img_idx = 0
for i in range(4):
    wr_en = 1
    wr_addr = img_idx
    wr_data = buf_rdata
    
    # Verilog parallel assignments using OLD values
    old_img_idx = img_idx
    
    img_idx = old_img_idx + 1
    if old_img_idx + 1 == img_size:
        state = "TX_SOF"
    else:
        buf_raddr = 4 + old_img_idx + 2  # <--- FIX: changed 1 to 2
    
    print(f"Write: addr={wr_addr}, data={wr_data:02X} (expected {payload[4+wr_addr]:02X})")
    posedge()
