#!/usr/bin/env python3
"""
UART loader for LeNet-5 on DE2.

This script sends model weights to external SRAM through a simple binary UART
protocol and can trigger inference afterwards.

Protocol summary (host -> device):
    [SOF=0xA5][CMD][SEQ][LEN_L][LEN_H][PAYLOAD...][CRC_L][CRC_H]

Protocol summary (device -> host):
    [SOF=0x5A][SEQ][STATUS][LEN_L][LEN_H][PAYLOAD...][CRC_L][CRC_H]

CRC is CRC-16/CCITT-FALSE over all bytes after SOF and before CRC.

Commands:
    0x00: PING
        Request payload: empty
        Response payload: optional ASCII text

    0x01: WRITE_SRAM
        Request payload: [ADDR_0][ADDR_1][ADDR_2][SIZE_L][SIZE_H][DATA...]
        Response payload: optional [SIZE_L][SIZE_H] echo

    0x02: START_INFERENCE
        Request payload: empty
        Response payload: empty

    0x03: GET_STATUS
        Request payload: empty
        Response payload (minimum 2 bytes): [DONE][DIGIT]

Device STATUS codes:
    0x00: OK
    non-zero: protocol or execution error

Note:
    The FPGA UART endpoint must implement the same frame format and commands.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import serial  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyserial. Install it with: pip install pyserial"
    ) from exc


HOST_SOF = 0xA5
DEV_SOF = 0x5A

CMD_PING = 0x00
CMD_WRITE_SRAM = 0x01
CMD_START_INFERENCE = 0x02
CMD_GET_STATUS = 0x03

# LeNet weight map in external SRAM.
WEIGHT_LAYOUT: Tuple[Tuple[str, str, int, int], ...] = (
    ("conv1", "conv1_weights.hex", 0, 150),
    ("conv3", "conv3_weights.hex", 150, 2400),
    ("c5", "c5_weights.hex", 2550, 30720),
    ("fc1", "fc1_weights.hex", 33270, 10080),
    ("fc2", "fc2_weights.hex", 43350, 840),
)


class ProtocolError(RuntimeError):
    """Raised when the UART protocol is violated."""


class DeviceStatusError(RuntimeError):
    """Raised when the device returns a non-zero status code."""


def crc16_ccitt_false(data: bytes) -> int:
    """Compute CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def parse_readmemh_bytes(path: Path) -> bytes:
    """Parse a Verilog $readmemh-style text file into raw bytes."""
    values: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            # Remove inline comment styles commonly used in hex files.
            line = raw_line.split("//", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue

            # Ignore optional @address markers if present.
            if line.startswith("@"):
                continue

            # Split by whitespace or commas.
            tokens = line.replace(",", " ").split()
            for token in tokens:
                token = token.strip().replace("_", "")
                if not token:
                    continue
                value = int(token, 16) & 0xFF
                values.append(value)

    return bytes(values)


class UartLeNetLoader:
    """Host-side UART transport for LeNet SRAM loading."""

    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self._seq = 0

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "UartLeNetLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFF
        return seq

    def _read_exact(self, nbytes: int) -> bytes:
        data = bytearray()
        while len(data) < nbytes:
            chunk = self.ser.read(nbytes - len(data))
            if not chunk:
                raise TimeoutError(
                    f"Timeout waiting for {nbytes} bytes from device")
            data.extend(chunk)
        return bytes(data)

    def _build_host_frame(self, cmd: int, seq: int, payload: bytes) -> bytes:
        if len(payload) > 0xFFFF:
            raise ValueError("Payload too large for protocol")

        header = bytes([HOST_SOF, cmd & 0xFF, seq & 0xFF]) + \
            len(payload).to_bytes(2, "little")
        crc = crc16_ccitt_false(header[1:] + payload)
        return header + payload + crc.to_bytes(2, "little")

    def _read_device_frame(self, expect_seq: int) -> bytes:
        # Byte-sync to SOF.
        while True:
            sof = self._read_exact(1)[0]
            if sof == DEV_SOF:
                break

        head = self._read_exact(4)
        seq = head[0]
        status = head[1]
        payload_len = int.from_bytes(head[2:4], "little")
        payload = self._read_exact(payload_len)
        crc_rx = int.from_bytes(self._read_exact(2), "little")

        crc_calc = crc16_ccitt_false(head + payload)
        if crc_rx != crc_calc:
            raise ProtocolError(
                f"CRC mismatch from device: rx=0x{crc_rx:04X}, calc=0x{crc_calc:04X}"
            )

        if seq != expect_seq:
            raise ProtocolError(
                f"Sequence mismatch: expected {expect_seq}, got {seq}")

        if status != 0:
            raise DeviceStatusError(f"Device returned STATUS=0x{status:02X}")

        return payload

    def transact(self, cmd: int, payload: bytes = b"") -> bytes:
        seq = self._next_seq()
        frame = self._build_host_frame(cmd, seq, payload)
        self.ser.write(frame)
        self.ser.flush()
        return self._read_device_frame(seq)

    def ping(self) -> str:
        payload = self.transact(CMD_PING)
        if not payload:
            return ""
        try:
            return payload.decode("ascii", errors="replace")
        except Exception:
            return payload.hex()

    def write_sram(self, addr: int, data: bytes) -> None:
        if addr < 0 or addr >= (1 << 24):
            raise ValueError(f"Address out of range: {addr}")
        if len(data) > 0xFFFF:
            raise ValueError(
                "Single WRITE_SRAM block is limited to 65535 bytes")

        payload = addr.to_bytes(3, "little") + \
            len(data).to_bytes(2, "little") + data
        ack = self.transact(CMD_WRITE_SRAM, payload)

        # Optional size echo check if device provides 2-byte payload.
        if len(ack) >= 2:
            written = int.from_bytes(ack[:2], "little")
            if written != len(data):
                raise ProtocolError(
                    f"Device wrote {written} bytes but host sent {len(data)} bytes"
                )

    def write_sram_chunked(self, base_addr: int, data: bytes, chunk_size: int, label: str) -> None:
        total = len(data)
        sent = 0
        start_t = time.time()

        for offset in range(0, total, chunk_size):
            chunk = data[offset: offset + chunk_size]
            self.write_sram(base_addr + offset, chunk)
            sent += len(chunk)

            if total <= chunk_size or (offset // chunk_size) % 16 == 0:
                percent = (100.0 * sent) / total if total else 100.0
                print(f"[{label}] {sent}/{total} bytes ({percent:5.1f}%)")

        elapsed = max(time.time() - start_t, 1e-6)
        rate = total / elapsed
        print(f"[{label}] done in {elapsed:.2f}s ({rate:.1f} B/s)")

    def start_inference(self) -> None:
        self.transact(CMD_START_INFERENCE)

    def get_status(self) -> Dict[str, int]:
        payload = self.transact(CMD_GET_STATUS)
        if len(payload) < 2:
            raise ProtocolError(
                "GET_STATUS response must contain at least 2 bytes: DONE and DIGIT")

        status: Dict[str, int] = {
            "done": payload[0],
            "digit": payload[1],
        }

        if len(payload) >= 3:
            status["busy"] = payload[2]

        return status


def collect_weight_segments(mem_dir: Path) -> List[Tuple[str, int, bytes]]:
    """Load and validate all configured model weight files."""
    segments: List[Tuple[str, int, bytes]] = []

    for name, filename, base, expected_size in WEIGHT_LAYOUT:
        path = mem_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing weight file: {path}")

        data = parse_readmemh_bytes(path)
        if len(data) != expected_size:
            raise ValueError(
                f"{filename}: expected {expected_size} bytes, got {len(data)} bytes"
            )

        segments.append((name, base, data))

    return segments


def wait_done(loader: UartLeNetLoader, timeout_sec: float, poll_interval: float) -> Dict[str, int]:
    """Poll GET_STATUS until done=1 or timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        st = loader.get_status()
        if st.get("done", 0):
            return st
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Inference did not finish within {timeout_sec:.1f} seconds")


def cmd_program(args: argparse.Namespace) -> int:
    mem_dir = Path(args.mem_dir).resolve()
    segments = collect_weight_segments(mem_dir)

    print(f"Using UART {args.port} @ {args.baud}")
    print(f"Memory folder: {mem_dir}")

    with UartLeNetLoader(args.port, args.baud, args.timeout) as loader:
        banner = loader.ping()
        if banner:
            print(f"Device banner: {banner}")
        else:
            print("Ping OK (no banner payload)")

        for name, base, data in segments:
            print(f"Programming {name}: base=0x{base:06X}, size={len(data)}")
            loader.write_sram_chunked(base, data, args.chunk_size, name)

        print("All weight segments programmed successfully.")

        if args.start:
            print("Sending START_INFERENCE command...")
            loader.start_inference()

            if args.wait_done:
                st = wait_done(loader, args.wait_timeout, args.poll_interval)
                print(
                    f"Inference done, predicted digit = {st.get('digit', 0)}")

    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    with UartLeNetLoader(args.port, args.baud, args.timeout) as loader:
        banner = loader.ping()
        print(f"Ping response: {banner if banner else '<empty>'}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    with UartLeNetLoader(args.port, args.baud, args.timeout) as loader:
        loader.start_inference()
        print("START_INFERENCE sent.")

        if args.wait_done:
            st = wait_done(loader, args.wait_timeout, args.poll_interval)
            print(f"Inference done, predicted digit = {st.get('digit', 0)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with UartLeNetLoader(args.port, args.baud, args.timeout) as loader:
        st = loader.get_status()
    print(
        f"done={st.get('done', 0)}, digit={st.get('digit', 0)}, busy={st.get('busy', 0)}")
    return 0


def cmd_write_file(args: argparse.Namespace) -> int:
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data = parse_readmemh_bytes(file_path)

    with UartLeNetLoader(args.port, args.baud, args.timeout) as loader:
        print(
            f"Programming {file_path.name} to 0x{args.addr:06X}, size={len(data)}")
        loader.write_sram_chunked(
            args.addr, data, args.chunk_size, file_path.name)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UART SRAM loader for LeNet-5 DE2")
    parser.add_argument("--port", required=True,
                        help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200,
                        help="UART baudrate (default: 115200)")
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="Serial timeout in seconds")

    sub = parser.add_subparsers(dest="command", required=True)

    p_program = sub.add_parser(
        "program", help="Program all model weight files to SRAM")
    p_program.add_argument(
        "--mem-dir",
        default=str(Path(__file__).resolve().parents[1] / "mem"),
        help="Folder containing weight hex files",
    )
    p_program.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Bytes per WRITE_SRAM command (default: 256)",
    )
    p_program.add_argument("--start", action="store_true",
                           help="Start inference after programming")
    p_program.add_argument("--wait-done", action="store_true",
                           help="Poll status until inference is done")
    p_program.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Timeout for wait-done in seconds",
    )
    p_program.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Polling interval for wait-done in seconds",
    )
    p_program.set_defaults(func=cmd_program)

    p_ping = sub.add_parser("ping", help="Check UART protocol connectivity")
    p_ping.set_defaults(func=cmd_ping)

    p_start = sub.add_parser("start", help="Send START_INFERENCE command")
    p_start.add_argument("--wait-done", action="store_true",
                         help="Poll status until inference is done")
    p_start.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Timeout for wait-done in seconds",
    )
    p_start.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Polling interval for wait-done in seconds",
    )
    p_start.set_defaults(func=cmd_start)

    p_status = sub.add_parser("status", help="Read done/digit status")
    p_status.set_defaults(func=cmd_status)

    p_write = sub.add_parser(
        "write-file", help="Program one readmemh file to SRAM")
    p_write.add_argument("--file", required=True,
                         help="Path to input .hex file")
    p_write.add_argument("--addr", required=True,
                         type=lambda x: int(x, 0), help="Base SRAM byte address")
    p_write.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Bytes per WRITE_SRAM command (default: 256)",
    )
    p_write.set_defaults(func=cmd_write_file)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except (ProtocolError, DeviceStatusError, TimeoutError, ValueError, FileNotFoundError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    except serial.SerialException as err:
        print(f"Serial error: {err}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
