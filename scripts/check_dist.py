"""Check a release's files before upload.

    python scripts/check_dist.py dist [v0.2.0]

Every file must carry one version (the tag's, when a tag is given, so a
dirty or untagged build can't ship as a release), there must be an sdist,
and there must be a wheel for every Python on every platform we build (see
[tool.cibuildwheel] in pyproject.toml).
"""
import re
import sys
from pathlib import Path

PYTHONS = ["cp310-cp310", "cp311-cp311", "cp312-cp312", "cp313-cp313", "cp314-cp314", "cp314-cp314t"]
PLATFORMS = ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64", "macosx_11_0_arm64", "macosx_11_0_x86_64"]


def main(dist, tag=None):
    files = sorted(p.name for p in Path(dist).iterdir())
    errors = []
    versions = set()
    for f in files:
        m = re.fullmatch(r"gpupipe-([^-]+)(?:\.tar\.gz|-.+\.whl)", f)
        if m is None:
            errors.append(f"unexpected file: {f}")
        else:
            versions.add(m.group(1))
    if len(versions) != 1:
        _fail(errors + [f"expected one version, found {sorted(versions)}"])
    version = versions.pop()
    if tag is not None and version != tag.removeprefix("v"):
        errors.append(f"files are version {version}, but the tag is {tag}")
    if f"gpupipe-{version}.tar.gz" not in files:
        errors.append("no sdist")
    for plat in PLATFORMS:
        for py in PYTHONS:
            if f"gpupipe-{version}-{py}-{plat}.whl" not in files:
                errors.append(f"missing wheel: {py} {plat}")
    if errors:
        _fail(errors)
    print(f"ok: gpupipe {version}, sdist + {len(files) - 1} wheels")


def _fail(errors):
    for e in errors:
        print(f"ERROR: {e}")
    sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])
