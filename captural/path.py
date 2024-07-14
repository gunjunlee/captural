import re
import os
import os.path as osp
from pathlib import Path

def _path_to_abs(path):
    path = Path(path).absolute().as_posix()
    return path

def _path_to_pattern(path):
    if path == ".":
        path = os.getcwd() + r'/.*'
    try:
        if Path(path).is_dir():
            path = osp.join(_path_to_abs(path), '.*')
        elif Path(path).is_file():
            path = _path_to_abs(path) + '$'
    except OSError:
        # If the path is not a valid path, we assume it is a regex
        pass
    try:
        path = str(path).replace(r'/', r'\/')
        return re.compile(path)
    except re.error:
        raise ValueError(f'Invalid path: {path}')


class PathChecker:
    def __init__(self, paths):
        self.patterns = [_path_to_pattern(path) for path in paths]

    def check(self, path):
        path = _path_to_abs(path)
        return any(p.match(path) for p in self.patterns)
