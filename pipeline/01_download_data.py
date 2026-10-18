"""Step 01: download INEC census data.

Fetches the 2022 manzana-level microdata, the 2010 microdata harmonized to
2022 geography, the variable dictionary, and the national geodatabase.
"""

import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from multiscalar_segregation import load_config, repo_root


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists, skipping: {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    mode = "wb"
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"
    with requests.get(url, stream=True, headers=headers, timeout=60, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + (tmp.stat().st_size if mode == "ab" else 0)
        with open(tmp, mode) as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc=dest.name, initial=f.tell()) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def main() -> None:
    cfg = load_config()
    raw = repo_root() / cfg["paths"]["data_raw"]
    raw.mkdir(parents=True, exist_ok=True)

    urls = [
        cfg["census"]["y2022"]["microdata_url"],
        cfg["census"]["y2022"]["dictionary_url"],
        cfg["census"]["y2010"]["microdata_url"],
        cfg["census"]["geodatabase_url"],
    ]
    for url in urls:
        fetch(url, raw / url.rsplit("/", 1)[1])

    sums_file = raw / "SHA256SUMS"
    known = {}
    if sums_file.exists():
        known = dict(line.split()[::-1] for line in sums_file.read_text().splitlines())
    lines = []
    for url in urls:
        name = url.rsplit("/", 1)[1]
        digest = sha256(raw / name)
        if name in known and known[name] != digest:
            sys.exit(f"CHECKSUM MISMATCH for {name}: expected {known[name]}, got {digest}")
        lines.append(f"{digest}  {name}")
    sums_file.write_text("\n".join(lines) + "\n")
    print("all checksums recorded/verified")


if __name__ == "__main__":
    main()
