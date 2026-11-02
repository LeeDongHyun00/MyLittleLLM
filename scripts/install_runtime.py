"""Install pinned official CPU binaries and GGUF weights, verifying SHA-256."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) == expected:
            print(f"Verified existing {path.name}", flush=True)
            return
        raise SystemExit(f"Existing file hash mismatch; inspect before replacing: {path}")
    partial = path.with_suffix(path.suffix + ".part")
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                    "--output", str(partial), url], check=True)
    if digest(partial) != expected:
        raise SystemExit(f"Downloaded checksum mismatch: {partial}")
    partial.replace(path)
    print(f"Verified {path.name}: {expected}", flush=True)


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This pinned binary targets Linux x86_64 only")
    lock = json.loads((ROOT / "config/runtime.lock.json").read_text())
    runtime = lock["llama_cpp"]
    archive = ROOT / "runtime" / (runtime["tag"] + ".tar.gz")
    download(runtime["url"], archive, runtime["sha256"])
    target = ROOT / "runtime" / runtime["tag"]
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    binaries = list(target.rglob("llama-server"))
    if len(binaries) != 1:
        raise SystemExit("Expected exactly one llama-server")
    (ROOT / "runtime/server_path.txt").write_text(str(binaries[0]) + "\n")
    model = lock["model"]
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['filename']}"
    download(url, ROOT / "models" / model["filename"], model["sha256"])
    print("Install complete. Run bash serve.sh", flush=True)


if __name__ == "__main__":
    main()
