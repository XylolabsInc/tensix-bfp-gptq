# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The calibration and evaluation texts, fetched once into a local directory.

  English   WikiText-2 (raw, Salesforce/wikitext, config wikitext-2-raw-v1) from the Hugging Face
            datasets server: rows of the train split for calibration, of the test split for
            evaluation, joined as they are
  Korean    plain-text extracts of Korean Wikipedia articles from the MediaWiki API (TextExtracts),
            two disjoint sets of titles for calibration and evaluation

  name        source                                              size
  calib_en    WikiText-2 train, rows from 0, 100 a request         until >= 30,000 characters
  eval_en     WikiText-2 test, rows from 0                          until >= 15,000 characters
  calib2_en   WikiText-2 train, rows from 1000                      until >= 110,000 characters
  calib_ko    KO_CALIB articles, characters 0 .. 3,500 of each      joined by a blank line
  eval_ko     KO_EVAL articles, characters 0 .. 3,500 of each       joined by a blank line
  calib2_ko   KO_CALIB articles, characters 3,500 .. 12,500 of each joined by a blank line

WikiText-2 is CC BY-SA 3.0 and GFDL, Korean Wikipedia CC BY-SA 4.0 (most text also GFDL); the
texts are not shipped with this package. The SHA-256 of each file as used for the published
exports is in SHA256: WikiText-2 fetches reproduce it, while Wikipedia articles change over time,
so a fresh Korean fetch may differ (and an export made from it too).

  python -m tensix_bfp.data fetch [--data DIR]    download the six texts (skips existing files)
  python -m tensix_bfp.data check [--data DIR]    compare the files with SHA256

DIR defaults to $TENSIX_BFP_DATA or ~/.cache/tensix_bfp/data.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "tensix-bfp-gptq/0.1 (calibration text fetch)"}
KO_CALIB = ["광합성", "한글", "김치", "도서관", "컴퓨터", "서울특별시", "태양계", "축구", "조선"]
KO_EVAL = ["세종 (조선)", "인공지능", "지구", "부산광역시", "커피", "고려", "피아노", "바다"]
CHARS = 3500  # per Korean article: its first paragraphs
SHA256 = {
    "calib_en": "88fa1a10cacc84b916b9d730259190ade358373a9797cfdb96049e17762529d6",
    "calib2_en": "db23f5c21d4e567433b63469178f2a63253c4299bad6acbf39ef19211368df11",
    "calib_ko": "e8c3413e6219c5504706c3e6e75d4661b13cee0bfb5a79d97e0d00b4d86c2304",
    "calib2_ko": "ac72a026bc596b9594a9f32525e39d7102e7d8e1700b9d718734f351165c09b8",
    "eval_en": "cb700e96eb44e3c39a4dd0cfd811896a7eef67cbf8c1d2c4b5739a3449a34386",
    "eval_ko": "f7b227bd69c84083ae51f456a1c38a859d3d3fa88416383e1d10349271e23328",
}


def default_dir():
    return Path(os.environ.get("TENSIX_BFP_DATA", Path.home() / ".cache/tensix_bfp/data"))


def get(url, tries=6):
    """JSON from url, retried with a growing pause on HTTP 429."""
    for n in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 429 or n == tries - 1:
                raise
            time.sleep(5 * (n + 1))
    return None


def wikitext(split, chars, off=0):
    """WikiText-2 raw rows of `split` from row `off`, 100 per request, until at least `chars`
    characters, joined without a separator."""
    out = []
    while sum(map(len, out)) < chars:
        q = urllib.parse.urlencode(
            {
                "dataset": "Salesforce/wikitext",
                "config": "wikitext-2-raw-v1",
                "split": split,
                "offset": off,
                "length": 100,
            }
        )
        rows = get(f"https://datasets-server.huggingface.co/rows?{q}")["rows"]
        if not rows:
            break
        out += [r["row"]["text"] for r in rows]
        off += 100
        time.sleep(0.5)
    return "".join(out)


def kowiki(titles, start=0, chars=CHARS):
    """Characters start .. start + chars of each Korean Wikipedia article's plain-text extract,
    joined by a blank line."""
    parts = []
    for t in titles:
        q = urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "redirects": 1,
                "titles": t,
            }
        )
        pages = get(f"https://ko.wikipedia.org/w/api.php?{q}")["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")
        print(f"  {t}: {len(text)} characters", flush=True)
        parts.append(text[start : start + chars])
        time.sleep(3)
    return "\n\n".join(parts)


JOBS = {
    "calib_en": lambda: wikitext("train", 30000),
    "eval_en": lambda: wikitext("test", 15000),
    "calib_ko": lambda: kowiki(KO_CALIB),
    "eval_ko": lambda: kowiki(KO_EVAL),
    "calib2_en": lambda: wikitext("train", 110000, off=1000),
    "calib2_ko": lambda: kowiki(KO_CALIB, start=CHARS, chars=9000),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        p = d / f"{name}.txt"
        if not p.exists():
            print(f"fetching {name}", flush=True)
            p.write_bytes(fn().encode("utf-8"))
        print(f"{name}: {len(p.read_text(encoding='utf-8'))} characters", flush=True)


def check(d):
    """Whether every text matches SHA256; prints one line per text."""
    ok = True
    for name, want in SHA256.items():
        p = Path(d) / f"{name}.txt"
        got = sha256(p) if p.exists() else None
        state = "missing" if got is None else ("ok" if got == want else "DIFFERS")
        ok &= state == "ok"
        print(f"{name}: {state}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["fetch", "check"])
    ap.add_argument("--data", default=None, help="the texts' directory")
    a = ap.parse_args(argv)
    d = Path(a.data) if a.data else default_dir()
    if a.what == "fetch":
        fetch(d)
        return 0 if check(d) else 1
    return 0 if check(d) else 1


if __name__ == "__main__":
    raise SystemExit(main())
