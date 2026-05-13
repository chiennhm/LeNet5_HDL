import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Add current directory to path so we can import uart_loader and uart_image_loader
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uart_loader import UartLeNetLoader
from uart_image_loader import write_image_by_command, write_image_by_sram

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Digit for LeNet-5 (DE2)")
        
        # LeNet-5 takes 28x28 images. We scale up for drawing.
        self.grid_size = 28
        self.pixel_size = 15
        self.canvas_size = self.grid_size * self.pixel_size
        
        # 2D array holding pixel intensity 0..255
        self.image_data = [[0] * self.grid_size for _ in range(self.grid_size)]
        
        # UI Setup
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_clear = tk.Button(btn_frame, text="Clear Canvas", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        self.btn_send = tk.Button(btn_frame, text="Send & Predict", command=self.send_and_predict, bg="green", fg="white", font=("Arial", 10, "bold"))
        self.btn_send.pack(side=tk.RIGHT, padx=5)
        
        # UART config frame
        config_frame = tk.Frame(root)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(config_frame, text="COM Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value="COM4")
        self.port_entry = tk.Entry(config_frame, textvariable=self.port_var, width=10)
        self.port_entry.pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = tk.Label(root, text="Status: Ready", fg="blue", font=("Arial", 12))
        self.lbl_status.pack(pady=10)
        
        self.baud = 115200
        self.method = "image-cmd"
        self.sram_base = 0x010000

    def paint(self, event):
        x, y = event.x, event.y
        grid_x = x // self.pixel_size
        grid_y = y // self.pixel_size
        
        # Draw a thick brush (center + neighbors)
        brush = [
            (0, 0, 255),
            (1, 0, 200), (-1, 0, 200), (0, 1, 200), (0, -1, 200),
            (1, 1, 150), (-1, -1, 150), (1, -1, 150), (-1, 1, 150)
        ]
        
        for dx, dy, intensity in brush:
            nx, ny = grid_x + dx, grid_y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                current = self.image_data[ny][nx]
                new_val = min(255, current + intensity)
                self.image_data[ny][nx] = new_val
                
                # Draw on canvas
                color = f"#{new_val:02x}{new_val:02x}{new_val:02x}"
                px1 = nx * self.pixel_size
                py1 = ny * self.pixel_size
                px2 = px1 + self.pixel_size
                py2 = py1 + self.pixel_size
                self.canvas.create_rectangle(px1, py1, px2, py2, fill=color, outline=color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_data = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.lbl_status.config(text="Status: Cleared", fg="blue")

    def print_ascii_art(self):
        print("Drawn Image:")
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                val = self.image_data[r][c]
                if val > 128:
                    row_str += "##"
                elif val > 64:
                    row_str += "::"
                elif val > 0:
                    row_str += ".."
                else:
                    row_str += "  "
            print(row_str)

    def send_and_predict(self):
        self.print_ascii_art()
        
        # Flatten image_data to 1D bytes
        flat_data = []
        for row in self.image_data:
            flat_data.extend(row)
        
        image_bytes = bytes(flat_data)
        port = self.port_var.get().strip()
        
        try:
            with UartLeNetLoader(port, self.baud, 1.0) as loader:
                banner = loader.ping()
                print(f"Connected. Device banner: {banner}")
                
                if self.method == "image-cmd":
                    write_image_by_command(loader, image_bytes, 128)
                else:
                    write_image_by_sram(loader, image_bytes, self.sram_base, 128)
                    
                print("Sending START_INFERENCE command...")
                loader.start_inference()
                
                self.lbl_status.config(text="Status: Inference running...", fg="orange")
                self.root.update()
                
                print("Waiting for result...")
                import time
                start_wait = time.time()
                digit = '?'
                while time.time() - start_wait < 5.0:
                    st = loader.get_status()
                    if st.get('done', 0) == 1:
                        digit = st.get('digit', '?')
                        break
                    time.sleep(0.05)
                
                self.lbl_status.config(text=f"Status: Predicted Digit = {digit}", fg="green")
                print(f"Inference complete. Predicted digit: {digit}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Communication failed:\n{str(e)}")
            self.lbl_status.config(text="Status: Error", fg="red")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Draw digit and send via UART for LeNet-5")
    parser.add_argument("--port", default="COM4", help="Serial port, e.g., COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    args = parser.parse_args()

    root = tk.Tk()
    app = DigitDrawer(root)
    app.port_var.set(args.port)
    app.baud = args.baud
    
    root.mainloop()

if __name__ == "__main__":
    main()
