"""Verify or explicitly refresh this public snapshot's SHA-256 inventory."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'SHA256SUMS'


def safe_file(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or '\n' in name or '\r' in name:
        raise ValueError('Unsafe inventory path: ' + repr(name))
    path = ROOT / name
    for part in [path, *path.parents]:
        if part == ROOT:
            break
        if part.is_symlink():
            raise ValueError('Symlink is not allowed: ' + name)
    if not path.resolve().is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Missing or invalid file: ' + name)
    return path


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def public_files():
    git = shutil.which('git')
    if git is None:
        return None
    # Isolate ignore rules from any parent repository, global excludes or Git history.
    with tempfile.TemporaryDirectory(prefix='rosa-publication-index-') as temporary:
        git_dir = Path(temporary) / 'index.git'
        subprocess.run([git, 'init', '--bare', '--quiet', str(git_dir)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p = subprocess.run([git, '--git-dir', str(git_dir), '--work-tree', str(ROOT),
                            '-c', 'core.excludesFile=/dev/null', 'ls-files', '--others',
                            '--exclude-standard', '-z'], check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, cwd=ROOT)
        return sorted(name for name in p.stdout.decode('utf-8').split('\0')
                      if name and name != MANIFEST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true',
                        help='Replace hashes after a separate privacy review; requires Git')
    args = parser.parse_args()
    actual = public_files()
    if args.refresh:
        if actual is None:
            raise ValueError('Git is required to refresh the inventory with .gitignore rules.')
        rows = [f'{sha(safe_file(name))}  {name}\n' for name in actual]
        (ROOT / MANIFEST).write_text(''.join(rows), encoding='utf-8')
        print(f'Updated {MANIFEST}: {len(rows)} files. This is not a privacy review.')
        return 0

    expected = {}
    for line in (ROOT / MANIFEST).read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if match is None:
            raise ValueError('Malformed checksum entry')
        value, name = match.groups()
        if name == MANIFEST or name in expected:
            raise ValueError('Duplicate or self-referential checksum entry: ' + name)
        expected[name] = value
    if not expected:
        raise ValueError('Empty inventory')
    failures = []
    for name, expected_hash in expected.items():
        try:
            if sha(safe_file(name)) != expected_hash:
                failures.append('Modified: ' + name)
        except ValueError as e:
            failures.append(str(e))
    if actual is not None:
        failures += ['Unexpected non-ignored file: ' + name for name in sorted(set(actual) - expected.keys())]
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f'PASS: {len(expected)} file hashes verified.')
    if actual is None:
        print('WARNING: Git is unavailable; unexpected non-ignored files were not checked.')
    else:
        print('PASS: no unexpected non-ignored files.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
