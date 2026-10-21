"""Download a verified MIT-licensed collection of real throughput traces."""
from __future__ import annotations
import shutil, urllib.request, zipfile
from pathlib import Path
URL="https://github.com/confiwent/Real-world-bandwidth-traces/archive/refs/heads/master.zip"
def main() -> None:
    destination=Path("data/raw/real_world_bandwidth_traces.zip"); destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        urllib.request.urlretrieve(URL,destination)
        with zipfile.ZipFile(destination) as archive:
            for member in archive.namelist():
                if any(part in member for part in ("/traces_3gp/","/fcc_ori/","/puffer_")) and not member.endswith("/"):
                    target=destination.parent / Path(member).name
                    with archive.open(member) as source, target.open("wb") as output: shutil.copyfileobj(source,output)
        print(f"Downloaded real traces to {destination.parent}")
    except Exception as error:
        print(f"Download failed: {error}\nManual download: {URL}\nPlace extracted two-column [timestamp_s throughput_mbps] files in data/raw/. The source repository is MIT licensed.")
if __name__ == "__main__": main()

