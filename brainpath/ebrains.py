"""Find and fetch atlas slides on the EBRAINS data proxy.

The atlas stores each slide as  v1.0/<diagnosis>/<uuid>.ndpi  in the bucket of
dataset 8fc108ab-e2b4-406f-8999-60269dc1f994, so a slide's location follows
from the annotation CSV. The data proxy hands out a short-lived download link
per file to a logged-in user; the login is an EBRAINS access token, read from
the EBRAINS_TOKEN environment variable.

Check that your token works on one slide before a long run:
  python -m brainpath.ebrains pilot_slides.csv
"""

import argparse
import os
from urllib.parse import quote

import pandas as pd

API = "https://data-proxy.ebrains.eu/api/v1/datasets"
ATLAS_DATASET = "8fc108ab-e2b4-406f-8999-60269dc1f994"
ATLAS_VERSION = "v1.0"


def object_name(tumour_type, slide_id, version=ATLAS_VERSION):
    name = str(slide_id)
    if not name.endswith(".ndpi"):
        name += ".ndpi"
    return f"{version}/{tumour_type}/{name}"


def auth_headers():
    token = os.environ.get("EBRAINS_TOKEN")
    if not token:
        raise SystemExit("set EBRAINS_TOKEN to your EBRAINS access token")
    return {"Authorization": f"Bearer {token}"}


def signed_url(obj, dataset=ATLAS_DATASET):
    """Ask the data proxy for a temporary download link to one file."""
    import requests

    r = requests.get(f"{API}/{dataset}/{quote(obj)}", params={"redirect": "false"},
                     headers=auth_headers(), timeout=60)
    if r.status_code in (401, 403):
        raise SystemExit(f"EBRAINS refused the token ({r.status_code}): log in again and refresh EBRAINS_TOKEN, "
                         "and check that you accepted the dataset's terms")
    r.raise_for_status()
    if "json" in r.headers.get("content-type", ""):
        return r.json()["url"]
    return r.url  # the proxy redirected straight to the file


def main():
    import requests

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="pilot_slides.csv from select_pilot")
    p.add_argument("--row", type=int, default=0, help="which slide to test")
    args = p.parse_args()

    row = pd.read_csv(args.manifest).iloc[args.row]
    obj = object_name(row.tumour_type, row.slide_id)
    print(f"slide: {obj}")
    url = signed_url(obj)
    with requests.get(url, headers={"Range": "bytes=0-1023"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        size = r.headers.get("content-range", "").split("/")[-1] or r.headers.get("content-length", "?")
    print(f"ok: download link works, file size {int(size) / 1e9:.2f} GB" if size.isdigit() else "ok: download link works")


if __name__ == "__main__":
    main()
