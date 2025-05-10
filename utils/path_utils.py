import os

def resolve_path(filename: str, subdir: str = ""):
    # Check if the file already exists in /tmp
    tmp_path = os.path.join("tmp", subdir, filename)
    repo_path = os.path.join("data", subdir, filename)

    # If the file exists in /tmp (user-generated), use it
    if os.path.exists(tmp_path):
        return tmp_path

    # Otherwise, fall back to the bundled repo version
    return repo_path 