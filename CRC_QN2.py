import csv
import sys

CRC_POLY = 0x4599
CRC_BITS = 15


def _int_to_bits(value: int, width: int) -> list:
    """Return `width` bits of `value`, MSB first."""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _bytes_to_bits(data) -> list:
    """Convert a sequence of bytes to a flat list of bits, MSB first per byte."""
    bits = []
    for byte in data:
        bits += _int_to_bits(byte, 8)
    return bits


def compute_crc15(bits: list) -> int:
    """Run the CAN CRC-15 shift register over a list of bits (0/1)."""
    crc = 0
    for bit in bits:
        top = (crc >> (CRC_BITS - 1)) & 1
        crc = ((crc << 1) | bit) & 0x7FFF
        if top:
            crc ^= CRC_POLY
    return crc


def compute_frame_crc(id_val: int, rtr: int, ide: int, dlc: int,
                      data_bytes: list) -> int:
    """
    Build the CAN frame bit sequence and return its CRC-15.

    Bit sequence:
        SOF(1b=0) | ID(11b) | RTR(1b) | IDE(1b) | r0(1b=0) | DLC(4b)
        | Data(DLC*8b) | 15 zero padding bits
    """
    bits = []
    bits.append(0)                             # SOF
    bits += _int_to_bits(id_val & 0x7FF, 11)  # 11-bit ID
    bits.append(rtr & 1)                       # RTR
    bits.append(ide & 1)                       # IDE
    bits.append(0)                             # r0 (reserved bit)
    bits += _int_to_bits(dlc & 0xF, 4)        # DLC
    bits += _bytes_to_bits(data_bytes)         # Data field
    bits += [0] * CRC_BITS                     # CRC field placeholder (15 zeros)
    return compute_crc15(bits)

#Frame Validation Logic

MAX_DLC     = 8
MAX_ID_11BIT = 0x7FF


def validate_frame(id_val, ide, rtr, dlc,
                   data_bytes, crc_stored):
    errors = []

    # 1. ID must fit in 11 bits (standard frame, IDE=0)
    if id_val > MAX_ID_11BIT:
        errors.append("bad_id")

    # 2. DLC must be 0–8
    if dlc > MAX_DLC:
        errors.append("bad_dlc")

    # 3. DLC must match actual data byte count
    if dlc <= MAX_DLC and len(data_bytes) != dlc:
        errors.append("mismatch_of_dlc_and_data_frame")

    # 4. CRC check (use at most dlc bytes for CRC if dlc is valid)
    crc_data    = data_bytes[:dlc] if dlc <= MAX_DLC else data_bytes
    crc_computed = compute_frame_crc(id_val, rtr, ide, dlc & 0xF, crc_data)
    if crc_computed != crc_stored:
        errors.append("bad_crc")

    return errors, crc_computed

# CSV processing and report

def parse_data_bytes(data_str: str) -> list:
    """Parse a space-separated hex byte string into a list of ints."""
    data_str = data_str.strip()
    return [int(b, 16) for b in data_str.split()] if data_str else []


def validate_csv(filepath: str):
    rows = []
    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    # Human-readable labels for each error code
    ERROR_LABELS = {
        "bad_id":                          "Bad CAN ID (exceeds 11 bits)",
        "bad_dlc":                         "Bad DLC (value > 8)",
        "mismatch_of_dlc_and_data_frame":  "Mismatch of DLC and length of Data",
        "bad_crc":                         "CRC mismatch",
    }



    for row in rows:
        id_val     = int(row["id"],  16)
        ide        = int(row["ide"])
        rtr        = int(row["rtr"])
        dlc        = int(row["dlc"])
        crc_stored = int(row["crc"], 16)
        data_bytes = parse_data_bytes(row["data"])
        given_err  = row["errors"].strip()

        errors, _ = validate_frame(id_val, ide, rtr, dlc, data_bytes, crc_stored)

        ts = row["timestamp"]

        if not errors:
            print(f"{ts}: The CAN frame check is success. "
                  f"(error: none). The given error is {given_err}")
        else:
            # Build a readable description of all detected errors
            readable = ", ".join(ERROR_LABELS.get(e, e) for e in errors)
            print(f"{ts}: The CAN frame check is failure. "
                  f"(error: {readable}). The given error is {given_err}")




if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "can_frames.csv"
    validate_csv(csv_path)