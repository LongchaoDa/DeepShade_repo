#!/usr/bin/env python3
import os
import re

# Directory containing the PNG files to rename
directory = "/scratch/YOURNAME/project/ControlNet_official_data/dataset/Tempe/source"

# Pattern matches filenames like:
#   3dmodelimage_6h_row3_col6.png
# Capturing:
#   hour   -> "6"
#   suffix -> "row3_col6.png"
pattern = re.compile(r'^3dmodelimage_(?P<hour>\d+)h_(?P<suffix>.+\.png)$')

for fname in os.listdir(directory):
    match = pattern.match(fname)
    if not match:
        continue

    hour = match.group('hour')
    suffix = match.group('suffix')

    # Build the new filename with 'newscreenshot' prefix
    new_name = f"newscreenshot_{hour}h_{suffix}"
    old_path = os.path.join(directory, fname)
    new_path = os.path.join(directory, new_name)

    print(f"Renaming {old_path} -> {new_path}")
    os.rename(old_path, new_path)

print("Done.")
