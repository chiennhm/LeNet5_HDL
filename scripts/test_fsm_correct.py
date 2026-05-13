import sys

payload = [0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x11, 0x22, 0x33, 0x44] # offset_L, offset_H, size_L, size_H, data0, data1, data2

buf_raddr = 0
buf_rdata = 0

state = "DISPATCH"
img_idx = 0
img_offset = 0
img_size = 0

def clock_edge():
    global buf_rdata, buf_raddr, state, img_idx, img_offset, img_size
    
    # Evaluate RHS using current states
    next_buf_rdata = payload[buf_raddr] if buf_raddr < len(payload) else 0xFF
    
    next_buf_raddr = buf_raddr
    next_state = state
    next_img_idx = img_idx
    next_img_offset = img_offset
    next_img_size = img_size
    
    wr_en = 0
    wr_addr = 0
    wr_data = 0
    
    if state == "DISPATCH":
        next_buf_raddr = 0
        next_state = "IMG_INIT"
        
    elif state == "IMG_INIT":
        next_buf_raddr = 1
        next_img_idx = 0
        next_state = "IMG_READ_HDR"
        
    elif state == "IMG_READ_HDR":
        if img_idx == 0:
            next_img_offset = (img_offset & 0xFF00) | buf_rdata
            next_buf_raddr = 2
        elif img_idx == 1:
            next_img_offset = (buf_rdata << 8) | (img_offset & 0xFF)
            next_buf_raddr = 3
        elif img_idx == 2:
            next_img_size = (img_size & 0xFF00) | buf_rdata
            next_buf_raddr = 4
        elif img_idx == 3:
            next_img_size = (buf_rdata << 8) | (img_size & 0xFF)
            next_img_idx = 0
            next_buf_raddr = 5  # Fixed: 4 -> 5
            next_state = "IMG_WRITE"
        if state == "IMG_READ_HDR":
            next_img_idx = img_idx + 1 if img_idx < 3 else 0
            
    elif state == "IMG_WRITE":
        wr_en = 1
        wr_addr = img_offset + img_idx
        wr_data = buf_rdata
        next_img_idx = img_idx + 1
        if img_idx + 1 == 4: # override img_size to 4 for short test
            next_state = "TX_SOF"
        else:
            next_buf_raddr = 4 + img_idx + 2  # Fixed: + 1 -> + 2
        
        print(f"Write: addr={wr_addr}, data={wr_data:02X}")
    
    # Apply updates
    buf_rdata = next_buf_rdata
    buf_raddr = next_buf_raddr
    state = next_state
    img_idx = next_img_idx
    img_offset = next_img_offset
    img_size = next_img_size

for i in range(15):
    clock_edge()

