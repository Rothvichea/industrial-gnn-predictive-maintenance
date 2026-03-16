"""
CWRU Bearing Dataset downloader.
10 classes: 1 normal + 3 fault types x 3 severities (007, 014, 021 inches)
All recorded at 12kHz, drive-end bearing, 1797 RPM load.
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Each entry: (local_filename, file_id_on_CWRU_server, label, fault_type, severity)
CWRU_FILES = [
    # Normal
    ("normal_1797.mat",   "97",  0, "normal",     0),
    # Inner race fault
    ("IR007_1797.mat",   "105",  1, "inner_race", 7),
    ("IR014_1797.mat",   "169",  2, "inner_race", 14),
    ("IR021_1797.mat",   "209",  3, "inner_race", 21),
    # Ball fault
    ("B007_1797.mat",    "118",  4, "ball",        7),
    ("B014_1797.mat",    "185",  5, "ball",       14),
    ("B021_1797.mat",    "222",  6, "ball",       21),
    # Outer race fault
    ("OR007_1797.mat",   "130",  7, "outer_race",  7),
    ("OR014_1797.mat",   "197",  8, "outer_race", 14),
    ("OR021_1797.mat",   "234",  9, "outer_race", 21),
]

BASE_URL = "https://engineering.case.edu/sites/default/files/"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_cwru(data_dir: str = "data/raw") -> dict:
    """
    Download all CWRU bearing files.
    Returns metadata dict for use in dataset loader.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    metadata = []
    already_have = 0

    for filename, file_id, label, fault_type, severity in CWRU_FILES:
        dest = Path(data_dir) / filename

        if dest.exists():
            print(f"  [skip] {filename} already exists")
            already_have += 1
        else:
            url = BASE_URL + file_id
            print(f"  [down] {filename}  (label={label}, {fault_type}, {severity}mil)")
            try:
                with DownloadProgressBar(unit='B', unit_scale=True,
                                         miniters=1, desc=filename, leave=False) as t:
                    urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
            except Exception as e:
                print(f"  [FAIL] {filename}: {e}")
                continue

        metadata.append({
            "path":       str(dest),
            "label":      label,
            "fault_type": fault_type,
            "severity":   severity,
            "filename":   filename,
        })

    print(f"\nDone. {already_have} skipped, {len(metadata)-already_have} downloaded.")
    print(f"Total files ready: {len(metadata)}/10")
    return metadata


if __name__ == "__main__":
    print("Downloading CWRU Bearing Dataset...")
    meta = download_cwru()
    print("\nMetadata sample:")
    for m in meta[:3]:
        print(" ", m)
