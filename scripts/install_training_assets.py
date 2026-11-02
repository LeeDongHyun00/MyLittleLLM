"""Download revision-pinned HF weights for unquantized training and compression."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "models/.hf-cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    from mylittlellm.distillation_data import sha256_file
    config = json.loads((ROOT / "config/distillation.json").read_text())
    manifest = []
    for repo, revision, folder in [(config['model'], config['revision'], config['local_model']),
                                   (config['compressor'], config['compressor_revision'], config['local_compressor'])]:
        path = Path(snapshot_download(repo, revision=revision, local_dir=ROOT / folder,
                                      allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE'], max_workers=2))
        manifest.append({"repo": repo, "revision": revision,
                         "files": [{"file": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)}
                                   for f in sorted(path.iterdir()) if f.is_file()]})
        print(repo, "downloaded and hashed", flush=True)
    output = ROOT / "outputs/training_assets.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
