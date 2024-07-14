import pytest

import tempfile
from pathlib import Path
from itertools import product

from captural import PathChecker

@pytest.mark.fast
@pytest.mark.parametrize(
    ("query", "keys"),
    product([
        "/a/b/c.py",
        "/a/b/c/d.py",
        "/a/b/c/d/e.py",
        "/a/b/c/d/e/f.py",
    ],
    [
        [
            "/a/b/c",
            "/a/c/d/c",
        ],
        [
            "/b/c",
            "/a/b/d/c",
        ],
        [
            "/a/b",
            "/a/b/c/d",
        ],
        [
            "/a/d",
            "/a/b/c/",
        ],
    ])
)
def test_path_plain(query, keys):
    assert PathChecker(keys).check(query) == any([
        query.startswith(key) for key in keys
    ])

@pytest.mark.fast
def test_path_re():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "a/b").mkdir(parents=True, exist_ok=True)
        (tmpdir / "a/b/c.py").touch()

        query = tmpdir / "a/b/c.py"

        keys = [
            tmpdir / "a/",
        ]
        assert PathChecker(keys).check(query) == True

        keys = [
            tmpdir / "a/.*",
        ]
        assert PathChecker(keys).check(query) == True

        keys = [
            tmpdir / "a/.*.py",
        ]
        assert PathChecker(keys).check(query) == True

        keys = [
            tmpdir / "a/.*.py$",
        ]
        assert PathChecker(keys).check(query) == True

        keys = [
            tmpdir / "b/",
        ]
        assert PathChecker(keys).check(query) == False

        keys = [
            tmpdir / "b/.*",
        ]
        assert PathChecker(keys).check(query) == False
