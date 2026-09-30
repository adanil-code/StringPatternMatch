#!/usr/bin/env python3
"""
Unicode Case-Folding Trie Generator (32-bit Delta Encoded)
----------------------------------------------------------
This script downloads the official UnicodeData.txt and generates a highly 
compressed, two-stage page-table trie for O(1) branchless case-folding.

It utilizes 32-bit Delta Encoding to safely bridge distant Unicode additions 
(such as U+019B -> U+A7AD, delta +42,561) while maintaining a strict 
branchless execution path.
"""

import urllib.request
import sys

def generate_trie():
    print("Downloading latest UnicodeData.txt...")
    url = "https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt"
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"Failed to download Unicode data: {e}")
        sys.exit(1)

    kMaxUnicode = 0x10FFFF
    kBlockSize = 256
    kDirectoryLength = (kMaxUnicode + kBlockSize) // kBlockSize

    upcase_delta_map = {}

    print("Parsing Unicode mappings...")
    # 1. Parse the mapping for cased characters
    for line in data:
        if not line:
            continue
            
        parts = line.split(';')
        code_point = int(parts[0], 16)
        uppercase_mapping = parts[12]
        
        if uppercase_mapping:
            upcased = int(uppercase_mapping, 16)
            delta = upcased - code_point
            upcase_delta_map[code_point] = delta

    # 2. Build the two-stage trie
    L1_Directory = [0] * kDirectoryLength
    L2_Data_Deltas = []

    # Block 0 is the Identity/Empty block (all zeroes, meaning no casing changes)
    L2_Data_Deltas.append([0] * kBlockSize)

    print("Compressing into two-stage page-table...")
    for block_index in range(kDirectoryLength):
        has_casing = False
        block_deltas = [0] * kBlockSize
        
        for offset in range(kBlockSize):
            cp = (block_index * kBlockSize) + offset
            
            # Fetch the delta, defaulting to 0 if the character has no uppercase mapping
            delta = upcase_delta_map.get(cp, 0)
            block_deltas[offset] = delta
            
            if delta != 0:
                has_casing = True
                
        if has_casing:
            # Register a new active block
            block_id = len(L2_Data_Deltas)
            L1_Directory[block_index] = block_id
            L2_Data_Deltas.append(block_deltas)

    # Calculate exact memory footprint for the header comments
    # L1 Directory: entries * 2 bytes (uint16_t)
    l1_kb = (kDirectoryLength * 2) / 1024.0
    # L2 Deltas: blocks * 256 entries * 4 bytes (int32_t)
    l2_kb = (len(L2_Data_Deltas) * kBlockSize * 4) / 1024.0
    total_kb = l1_kb + l2_kb

    # 3. Emit the C++ Header
    out_filename = "unicode_trie_data.h"
    print(f"Writing C++ structures to {out_filename}...")
    
    with open(out_filename, "w") as f:
        f.write("/*\n")
        f.write(" * Auto-generated 32-bit Delta-Encoded Unicode Trie\n")
        f.write(" * Generated from official UnicodeData.txt\n")
        f.write(" * \n")
        f.write(f" * Total Footprint: ~{total_kb:.1f} KB (Fits entirely in standard L1/L2 Cache)\n")
        f.write(f" * L1 Directory: 4352 entries * 2 bytes = {l1_kb:.1f} KB\n")
        f.write(f" * L2 Deltas:    {len(L2_Data_Deltas)} blocks * 256 entries * 4 bytes = {l2_kb:.1f} KB\n")
        f.write(" */\n\n")
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        
        # Write Level 1 Directory (uint16_t)
        f.write(f"inline constexpr uint16_t L1_Directory[{kDirectoryLength}] = {{\n    ")
        for i, val in enumerate(L1_Directory):
            f.write(f"{val}")
            if i < len(L1_Directory) - 1:
                f.write(", ")
            # Line break every 16 items for readability
            if (i + 1) % 16 == 0 and i < len(L1_Directory) - 1:
                f.write("\n    ")
        f.write("\n};\n\n")
        
        # Write Level 2 Data Deltas (int32_t)
        f.write(f"inline constexpr int32_t L2_Data_Deltas[{len(L2_Data_Deltas)}][{kBlockSize}] = {{\n")
        for block_idx, block in enumerate(L2_Data_Deltas):
            f.write(f"    // Block {block_idx}\n    {{ ")
            for i, val in enumerate(block):
                f.write(f"{val}")
                if i < len(block) - 1:
                    f.write(", ")
            f.write(" }")
            if block_idx < len(L2_Data_Deltas) - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n")

    print(f"Success! Compressed Unicode mapping into {len(L2_Data_Deltas)} active data blocks.")
    print(f"Total C++ array footprint: {total_kb:.1f} KB.")

if __name__ == "__main__":
    generate_trie()