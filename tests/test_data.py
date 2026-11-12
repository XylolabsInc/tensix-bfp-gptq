# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The data module's bookkeeping, without network: every fetch job has a recorded hash, the extra
texts stay out of the default fetch and check, and every config names only known texts."""

import tomllib
from pathlib import Path

from tensix_bfp import data

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_jobs_and_hashes_match():
    assert set(data.JOBS) == set(data.SHA256)
    assert set(data.JOBS_EXTRA) == set(data.SHA256_EXTRA)
    assert not set(data.JOBS) & set(data.JOBS_EXTRA)


def test_extra_articles_are_new():
    assert not set(data.KO_CALIB3) & (set(data.KO_CALIB) | set(data.KO_EVAL))


def test_check_defaults_to_the_six(tmp_path, capsys):
    assert not data.check(tmp_path)
    out = capsys.readouterr().out
    assert "calib3_en" not in out and "calib_en: missing" in out
    assert not data.check(tmp_path, extra=True)
    assert "calib3_ko: missing" in capsys.readouterr().out


def test_configs_name_known_texts():
    known = set(data.SHA256) | set(data.SHA256_EXTRA)
    for c in CONFIGS.glob("*.toml"):
        cfg = tomllib.loads(c.read_text())
        assert set(cfg.get("calib", {}).get("texts", [])) <= known, c.name
