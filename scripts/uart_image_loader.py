#!/usr/bin/env python3
"""
UART image loader for LeNet-5 inference on DE2.

This script is intended for runtime image updates (per inference) and reuses the
transport implementation from scripts/uart_loader.py.

Supported upload methods:
1) image-cmd (default)
   Sends CMD_WRITE_IMAGE chunks to a dedicated image buffer in FPGA logic.

2) sram
   Writes image bytes to external SRAM using existing WRITE_SRAM command.
   Useful when firmware maps inference input image in SRAM space.

Expected image format:
- Verilog readmemh style, 1 byte per pixel, total 784 bytes (28x28).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uart_loader import (
    DeviceStatusError,
    MAX_DEVICE_PAYLOAD,
    MAX_SRAM_CHUNK,
    ProtocolError,
    UartLeNetLoader,
    parse_readmemh_bytes,
    wait_done,
)

# New command for direct image streaming endpoint in FPGA UART firmware.
CMD_WRITE_IMAGE = 0x04
IMAGE_SIZE = 784
MAX_IMAGE_CHUNK = MAX_DEVICE_PAYLOAD - 4


def load_image_bytes(image_file: Path) -> bytes:
    """Load and validate a 28x28 grayscale image from readmemh text file."""
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file}")

    data = parse_readmemh_bytes(image_file)
    if len(data) != IMAGE_SIZE:
        raise ValueError(
            f"Image size must be {IMAGE_SIZE} bytes (28x28), got {len(data)} bytes"
        )

    return data


def write_image_by_command(
    loader: UartLeNetLoader,
    image_data: bytes,
    chunk_size: int,
) -> None:
    """
    Write image bytes using CMD_WRITE_IMAGE.

    Payload format:
        [OFFSET_L][OFFSET_H][SIZE_L][SIZE_H][DATA...]
    """
    if chunk_size <= 0 or chunk_size > MAX_IMAGE_CHUNK:
        raise ValueError(
            f"image-cmd chunk_size must be in range 1..{MAX_IMAGE_CHUNK}")

    sent = 0
    total = len(image_data)

    for offset in range(0, total, chunk_size):
        chunk = image_data[offset: offset + chunk_size]
        payload = offset.to_bytes(2, "little") + \
            len(chunk).to_bytes(2, "little") + chunk
        ack = loader.transact(CMD_WRITE_IMAGE, payload)

        # Optional 2-byte ack for accepted size.
        if len(ack) >= 2:
            accepted = int.from_bytes(ack[:2], "little")
            if accepted != len(chunk):
                raise ProtocolError(
                    f"Device accepted {accepted} bytes but host sent {len(chunk)} bytes"
                )

        sent += len(chunk)
        percent = (100.0 * sent) / total
        print(f"[image-cmd] {sent}/{total} bytes ({percent:5.1f}%)")


def write_image_by_sram(
    loader: UartLeNetLoader,
    image_data: bytes,
    sram_base: int,
    chunk_size: int,
) -> None:
    """Write image bytes directly to SRAM address space using WRITE_SRAM."""
    print(f"[sram] base=0x{sram_base:06X}, size={len(image_data)}")
    loader.write_sram_chunked(sram_base, image_data, chunk_size, "image")


def run(args: argparse.Namespace) -> int:
    image_file = Path(args.image).resolve()
    image_data = load_image_bytes(image_file)

    print(f"Using image file: {image_file}")
    print(f"Using UART {args.port} @ {args.baud}")

    with UartLeNetLoader(args.port, args.baud, args.timeout) as loader:
        banner = loader.ping()
        if banner:
            print(f"Device banner: {banner}")
        else:
            print("Ping OK (no banner payload)")

        if args.method == "image-cmd":
            write_image_by_command(loader, image_data, args.chunk_size)
        else:
            write_image_by_sram(loader, image_data,
                                args.sram_base, args.chunk_size)

        if args.start:
            print("Sending START_INFERENCE command...")
            loader.start_inference()

            if args.wait_done:
                st = wait_done(loader, args.wait_timeout, args.poll_interval)
                print(
                    f"Inference done, predicted digit = {st.get('digit', 0)}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UART runtime image loader for LeNet-5")
    parser.add_argument("--port", required=True,
                        help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200,
                        help="UART baudrate")
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="Serial timeout in seconds")

    parser.add_argument(
        "--image",
        default=str("D:\\LeNetHDL\\scripts\\hex_images\\test_img_1_label_2.hex"),  
        help="Path to 28x28 image in readmemh format",
    )
    parser.add_argument(
        "--method",
        choices=("image-cmd", "sram"),
        default="image-cmd",
        help="Upload method: dedicated image command or direct SRAM write",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Chunk size in bytes per UART transaction",
    )
    parser.add_argument(
        "--sram-base",
        type=lambda x: int(x, 0),
        default=0x50000,
        help="SRAM base address used only when --method sram",
    )

    parser.add_argument("--start", action="store_true",
                        help="Start inference after image upload")
    parser.add_argument("--wait-done", action="store_true",
                        help="Wait for inference done status")
    parser.add_argument("--wait-timeout", type=float,
                        default=10.0, help="Wait timeout in seconds")
    parser.add_argument("--poll-interval", type=float,
                        default=0.05, help="Status poll interval")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    max_chunk = MAX_IMAGE_CHUNK if args.method == "image-cmd" else MAX_SRAM_CHUNK
    if args.chunk_size <= 0 or args.chunk_size > max_chunk:
        print(
            f"ERROR: --chunk-size must be in range 1..{max_chunk} for {args.method}",
            file=sys.stderr)
        return 2

    try:
        return int(run(args))
    except (ProtocolError, DeviceStatusError, TimeoutError, ValueError, FileNotFoundError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
