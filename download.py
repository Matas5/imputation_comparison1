from pathlib import Path
from urllib.request import urlretrieve
import tarfile

URL = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"

RAW_DIR = Path("data/raw/physio2012")
ARCHIVE = RAW_DIR / "set-a.tar.gz"
EXTRACTED_DIR = RAW_DIR / "set-a"


def download_physio2012():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACTED_DIR.exists():
        print("PhysioNet 2012 already downloaded.")
        return

    if not ARCHIVE.exists():
        print("Downloading PhysioNet 2012...")
        urlretrieve(URL, ARCHIVE)

    print("Extracting PhysioNet 2012...")

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(RAW_DIR, filter="data")

    print("Finished.")


if __name__ == "__main__":
    download_physio2012()