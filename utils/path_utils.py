import os
import tempfile

def resolve_path(filename: str, subdir: str = "", write_mode: bool = False):
    tmp_root = tempfile.gettempdir()
    tmp_path = os.path.normpath(os.path.join(tmp_root, subdir, filename))
    repo_path = os.path.normpath(os.path.join("data", subdir, filename))

    # Debug
    print(f"🔍 Checking temp path: {tmp_path} (exists={os.path.isfile(tmp_path)})")
    print(f"🔍 Checking data path: {repo_path} (exists={os.path.isfile(repo_path)})")

    if write_mode:
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        return tmp_path

    if os.path.isfile(tmp_path):
        return tmp_path

    if os.path.isfile(repo_path):
        return repo_path

    raise FileNotFoundError(
        f"File not found in either path:\n  - temp: {tmp_path}\n  - fallback: {repo_path}"
    )
