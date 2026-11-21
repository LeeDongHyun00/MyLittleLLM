"""Run a command with the rule engine loading a given operator library instead of the shipped manifest.

The engine reads data/rule_pilot/operator_packs.json unless RULE_PACKS overrides it (backinfer.load_packs);
an override skips the manifest's retirements and search features, so this wrapper passes all three from
the given manifest. It lets a candidate library be measured before it is shipped.

  python scripts/with_library.py outputs/rule_round4/manifest_candidate.json -- \\
      python scripts/seedlm4_tasks.py holdout --model s5-var --suffix _r4
"""
import json
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit(__doc__)
    manifest = json.loads(Path(sys.argv[1]).read_text())
    environment = dict(os.environ, RULE_PACKS=",".join(manifest.get("active", [])) or "none",
                       RULE_RETIRE=",".join(manifest.get("retired", {})), RULE_ENGINE=",".join(manifest.get("engine", [])))
    os.execvpe(sys.argv[3], sys.argv[3:], environment)


if __name__ == "__main__":
    main()
