"""Install the pinned runtime, dataset and student models for the rule-distillation pilot."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from install_runtime import ROOT, download  # reuse the checksum-verifying downloader

LOCK = ROOT / "config/rule_pilot.lock.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["coder-1.5b"],
                        help="keys from config/rule_pilot.lock.json; 'all' installs every pinned model")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
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
    for item in lock["datasets"].values():
        download(item["url"], ROOT / "data/raw/rule_pilot" / item["filename"], item["sha256"])
    wanted = list(lock["models"]) if args.models == ["all"] else args.models
    for key in wanted:
        if key not in lock["models"]:
            raise SystemExit(f"Unknown model key: {key}")
        model = lock["models"][key]
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['filename']}"
        download(url, ROOT / "models" / model["filename"], model["sha256"])
    print("Install complete", flush=True)


if __name__ == "__main__":
    main()
