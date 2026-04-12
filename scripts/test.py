import pandas as pd
import io
from PIL import Image
import numpy as np
import os
import argparse

def extract_parquet_to_hex(parquet_path, output_dir, num_images=5):
    print(f"Reading dataset: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the first `num_images` images
    for i in range(min(num_images, len(df))):
        row = df.iloc[i]
        label = row['label']
        img_dict = row['image']
        
        # HuggingFace Datasets 'image' column struct has a 'bytes' field
        img_bytes = img_dict['bytes']
        
        # Load image from bytes
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to grayscale numpy array
        img_np = np.array(img.convert('L'))
        
        # Ensure it's 28x28 (MNIST standard)
        if img_np.shape != (28, 28):
            raise ValueError(f"Expected 28x28 image, got {img_np.shape}")
        
        # Generate hex string
        hex_filename = os.path.join(output_dir, f"test_img_{i}_label_{label}.hex")
        
        with open(hex_filename, "w") as f:
            # Flatten array and write each pixel as 2-digit hex
            for pixel in img_np.flatten():
                f.write(f"{pixel:02X}\n")
                
        print(f"Saved {hex_filename} (Label: {label})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet images to hex.")
    parser.add_argument("--input", type=str, default=r"d:\LeNetHDL\scripts\test-00000-of-00001.parquet", help="Path to parquet file")
    parser.add_argument("--outdir", type=str, default=r"d:\LeNetHDL\scripts\hex_images", help="Output directory")
    parser.add_argument("--num", type=int, default=5, help="Number of images to convert")
    
    args = parser.parse_args()
    
    extract_parquet_to_hex(args.input, args.outdir, args.num)
    print("Done!")