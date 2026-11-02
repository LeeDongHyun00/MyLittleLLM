"""Read-only check for the user's OpenAI-compatible model endpoint."""
import argparse
import json
import urllib.request
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
args = parser.parse_args()
url = args.base_url.rstrip("/") + "/models"
opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
          if urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}
          else urllib.request.build_opener())
try:
    with opener.open(url, timeout=10) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
except Exception as error:
    raise SystemExit(f"Model endpoint unavailable: {type(error).__name__}: {error}")
