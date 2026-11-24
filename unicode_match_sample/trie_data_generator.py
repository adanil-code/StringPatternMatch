#!/usr/bin/env python3
"""
Unicode Simple-Upcasing Trie Generator (Delta Encoded)
------------------------------------------------------
This script downloads the pinned official UnicodeData.txt and generates a highly
compressed, two-stage page-table trie for O(1) branchless simple upcasing.

On 32-bit wchar_t platforms (Linux/macOS), it utilizes 32-bit signed Delta
Encoding to safely bridge distant Unicode additions (such as U+019B -> U+A7AD,
delta +42,514) while maintaining a strict branchless execution path.

On 16-bit wchar_t platforms (Windows, WCHAR_MAX <= 0xFFFF), it emits a compact
BMP-only table using 16-bit unsigned wrap-around deltas, avoiding unreachable
supplementary-plane entries.
"""

import argparse
from pathlib import Path
import sys
import urllib.request

kDefaultUnicodeVersion = "16.0.0"
kMaxBmp = 0xFFFF
kMaxUnicode = 0x10FFFF
kBlockSize = 256


def build_trie(upcase_delta_map, max_code_point, wrap_u16):
    directory_length = (max_code_point + kBlockSize) // kBlockSize

    # 2. Build the two-stage trie
    l1_directory = [0] * directory_length
    l2_data_deltas = []

    # Block 0 is the Identity/Empty block (all zeroes, meaning no casing changes)
    l2_data_deltas.append([0] * kBlockSize)

    for block_index in range(directory_length):
        has_casing = False
        block_deltas = [0] * kBlockSize

        for offset in range(kBlockSize):
            cp = (block_index * kBlockSize) + offset

            # Fetch the delta, defaulting to 0 if the character has no uppercase mapping
            delta = upcase_delta_map.get(cp, 0)

            if delta != 0:
                if wrap_u16:
                    target = cp + delta
                    if not (0 <= target <= kMaxBmp):
                        raise ValueError(
                            f"BMP code point U+{cp:04X} maps outside BMP to U+{target:04X}; "
                            "cannot encode safely in 16-bit wchar_t table."
                        )
                    block_deltas[offset] = delta & 0xFFFF
                else:
                    block_deltas[offset] = delta
                has_casing = True

        if has_casing:
            # Register a new active block
            block_id = len(l2_data_deltas)
            l1_directory[block_index] = block_id
            l2_data_deltas.append(block_deltas)

    if len(l2_data_deltas) > 256:
        raise ValueError(
            f"Active L2 block count ({len(l2_data_deltas)}) exceeds uint8_t capacity (256)."
        )

    return l1_directory, l2_data_deltas


def write_l1_directory(f, l1_directory):
    f.write(f"inline constexpr uint8_t L1_Directory[{len(l1_directory)}] = {{\n    ")
    for i, val in enumerate(l1_directory):
        f.write(f"{val}")
        if i < len(l1_directory) - 1:
            f.write(", ")
        # Line break every 16 items for readability
        if (i + 1) % 16 == 0 and i < len(l1_directory) - 1:
            f.write("\n    ")
    f.write("\n};\n\n")


def write_l2_deltas(f, l2_data_deltas, element_type):
    f.write(f"inline constexpr {element_type} L2_Data_Deltas[{len(l2_data_deltas)}][{kBlockSize}] = {{\n")
    for block_idx, block in enumerate(l2_data_deltas):
        f.write(f"    // Block {block_idx}\n    {{ ")
        for i, val in enumerate(block):
            f.write(f"{val}")
            if i < len(block) - 1:
                f.write(", ")
        f.write(" }")
        if block_idx < len(l2_data_deltas) - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n")


def generate_trie():
    parser = argparse.ArgumentParser(description="Generate Unicode simple-upcasing trie header.")
    parser.add_argument("--version", default=kDefaultUnicodeVersion, help="Pinned Unicode version")
    parser.add_argument("--input", default=None, help="Optional local path to UnicodeData.txt")
    parser.add_argument("--output", default="unicode_trie_data.h", help="Output C++ header path")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP request timeout in seconds")
    args = parser.parse_args()

    if args.input:
        print(f"Loading local UnicodeData.txt from {args.input}...")
        data = Path(args.input).read_text(encoding="utf-8").splitlines()
    else:
        url = f"https://www.unicode.org/Public/{args.version}/ucd/UnicodeData.txt"
        print(f"Downloading Unicode {args.version} UnicodeData.txt...")
        try:
            with urllib.request.urlopen(url, timeout=args.timeout) as response:
                data = response.read().decode("utf-8").splitlines()
        except Exception as e:
            print(f"Failed to download Unicode data: {e}", file=sys.stderr)
            sys.exit(1)

    upcase_delta_map = {}

    print("Parsing Unicode mappings...")
    # 1. Parse the mapping for cased characters
    for line in data:
        if not line or line.startswith("#"):
            continue

        parts = line.split(";")
        code_point = int(parts[0], 16)
        uppercase_mapping = parts[12]

        if uppercase_mapping:
            upcased = int(uppercase_mapping, 16)
            delta = upcased - code_point
            if delta != 0:
                upcase_delta_map[code_point] = delta

    print("Compressing into two-stage page-table...")
    bmp_l1, bmp_l2 = build_trie(upcase_delta_map, kMaxBmp, wrap_u16=True)
    full_l1, full_l2 = build_trie(upcase_delta_map, kMaxUnicode, wrap_u16=False)

    # Calculate exact memory footprints for the header comments
    # BMP (16-bit wchar_t): L1 uint8_t + L2 uint16_t wrap-around deltas
    bmp_l1_kb = len(bmp_l1) / 1024.0
    bmp_l2_kb = (len(bmp_l2) * kBlockSize * 2) / 1024.0
    bmp_total_kb = bmp_l1_kb + bmp_l2_kb

    # Full (32-bit wchar_t): L1 uint8_t + L2 int32_t signed deltas
    full_l1_kb = len(full_l1) / 1024.0
    full_l2_kb = (len(full_l2) * kBlockSize * 4) / 1024.0
    full_total_kb = full_l1_kb + full_l2_kb

    # 3. Emit the C++ Header
    out_filename = args.output
    print(f"Writing C++ structures to {out_filename}...")

    with open(out_filename, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * Auto-generated Delta-Encoded Unicode Simple-Upcasing Trie\n")
        f.write(f" * Generated from official Unicode {args.version} UnicodeData.txt\n")
        f.write(" *\n")
        f.write(f" * 16-bit wchar_t (BMP) Footprint:  ~{bmp_total_kb:.1f} KB\n")
        f.write(f" *   L1 Directory: {len(bmp_l1)} entries * 1 byte  = {bmp_l1_kb:.2f} KB\n")
        f.write(f" *   L2 Deltas:    {len(bmp_l2)} blocks * {kBlockSize} entries * 2 bytes = {bmp_l2_kb:.1f} KB\n")
        f.write(" *\n")
        f.write(f" * 32-bit wchar_t (Full) Footprint: ~{full_total_kb:.1f} KB\n")
        f.write(f" *   L1 Directory: {len(full_l1)} entries * 1 byte  = {full_l1_kb:.2f} KB\n")
        f.write(f" *   L2 Deltas:    {len(full_l2)} blocks * {kBlockSize} entries * 4 bytes = {full_l2_kb:.1f} KB\n")
        f.write(" */\n\n")
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n")
        f.write("#include <cwchar>\n\n")
        f.write("namespace unicode_case_detail\n{\n\n")
        f.write("#if WCHAR_MAX <= 0xFFFF\n\n")

        # Write Level 1 Directory (uint8_t) & Level 2 Data Deltas (uint16_t wrap-around) for BMP
        write_l1_directory(f, bmp_l1)
        write_l2_deltas(f, bmp_l2, "uint16_t")

        f.write("\n#else\n\n")

        # Write Level 1 Directory (uint8_t) & Level 2 Data Deltas (int32_t) for Full Unicode
        write_l1_directory(f, full_l1)
        write_l2_deltas(f, full_l2, "int32_t")

        f.write("\n#endif\n\n")
        f.write("} // namespace unicode_case_detail\n")

    print(
        f"Success! Compressed Unicode mapping into {len(bmp_l2)} BMP blocks (~{bmp_total_kb:.1f} KB) "
        f"and {len(full_l2)} Full blocks (~{full_total_kb:.1f} KB)."
    )


if __name__ == "__main__":
    generate_trie()