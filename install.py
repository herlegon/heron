import argparse
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
HINSTALL_GIT = "git+https://github.com/herlegon/hinstall.git"
HINSTALL_DEV = ROOT.parent / "hinstall"


def update_pip(python_exe: Path | str | None = None, timeout: float = 120) -> bool:
    python_exe = str(sys.executable if python_exe is None else python_exe)
    # Liste d'arguments plutôt que split(' ') : robuste aux chemins avec espaces
    # (ex. "C:\Program Files\Python313\python.exe")
    pip_command = [python_exe, "-m", "pip", "install", "--upgrade", "pip"]
    try:
        result = subprocess.run(
            pip_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while updating pip: {e}\n{e.stdout}")
        return False
    except Exception as e:
        print(f"Unexpected error while updating pip: {e}")
        return False

    print(result.stdout.strip())
    return True



def run(*args) -> None:
    cmd = [sys.executable, *map(str, args)]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd)



def pip(*args) -> None:
    run("-m", "pip", *args)



def installed_git_commits() -> dict[str, str]:
    """{url_git: commit_id} pour tous les packages installés depuis un dépôt git."""
    result = {}
    for dist in md.distributions():
        raw = dist.read_text("direct_url.json")
        if not raw:
            continue
        try:
            info = json.loads(raw)
        except ValueError:
            continue
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit and "url" in info:
            result[info["url"].removesuffix(".git")] = commit
    return result



def remote_commit(url: str, ref: str = "HEAD") -> str | None:
    try:
        out = subprocess.run(
            ["git", "ls-remote", url, ref],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception:
        return None
    return out.split()[0] if out.strip() else None



def split_git_requirement(line: str) -> tuple[str, str]:
    """'git+https://x/y.git@branch' -> ('https://x/y.git', 'branch')"""
    url = line.removeprefix("git+")
    head, _, tail = url.rpartition("/")
    if "@" in tail:
        tail, _, ref = tail.partition("@")
        return f"{head}/{tail}", ref
    return url, "HEAD"



def is_git_up_to_date(line: str, installed: dict[str, str]) -> bool:
    url, ref = split_git_requirement(line)
    local = installed.get(url.removesuffix(".git"))
    if local is None:
        return False
    remote = remote_commit(url, ref)
    return remote is not None and remote == local



_PATH_OPT_RE = re.compile(r"^(-r|--requirement|-c|--constraint|-e|--editable)[\s=]+(\S.*)$")
def _absolutize(line: str, base: Path) -> str:
    # only local pathes become absolute
    m = _PATH_OPT_RE.match(line.strip())
    if not m:
        return line
    opt, value = m.groups()
    # Do not modify "-e git+https://..."
    if "://" not in value:
        p = Path(value)
        if not p.is_absolute():
            p = base / p
        value = p.resolve().as_posix()
    return f"{opt} {value}"



def filter_requirements(req_file: Path, installed: dict[str, str], out_dir: Path) -> Path:
    kept = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("git+") and is_git_up_to_date(line, installed):
            print(f"  = {line} : already up to date, ignoring")
        else:
            kept.append(_absolutize(raw, req_file.parent))

    out = out_dir / f"{req_file.stem}.filtered.txt"
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return out



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upgrade", action="store_true")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--toml", default=str(ROOT / "packages.toml"))
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    devmode: bool = args.dev

    # Try to upgrade pip
    if args.upgrade and not update_pip():
        print("Failed to upgrade pip.")

    # Install hinstall
    if devmode and HINSTALL_DEV.exists():
        pip("install", "-e", HINSTALL_DEV)
    elif args.force or not is_git_up_to_date(HINSTALL_GIT, installed_git_commits()):
        extra = ["--force-reinstall"] if args.upgrade else []
        pip("install", "--no-deps", *extra, HINSTALL_GIT)
    else:
        print("hinstall is already up-to-date")

    # python package requirements
    req = ROOT / ("requirements-dev.txt" if devmode else "requirements.txt")
    installed = {} if args.force else installed_git_commits()
    with tempfile.TemporaryDirectory(prefix="install_") as tmp:
        filtered = filter_requirements(req, installed, Path(tmp))
        pip("install", *(["--upgrade"] if args.upgrade else []), "-r", filtered)


    # install packages
    importlib.invalidate_caches()
    from hinstall import (
        install_app_packages,
        ilog,
    )

    if args.debug:
        ilog.setLevel("DEBUG")
    else:
        ilog.setLevel("INFO")

    ok = install_app_packages(
        args.toml,
        reinstall=args.reinstall,
        verbose=args.verbose
    )
    return 0 if ok else 1



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()


