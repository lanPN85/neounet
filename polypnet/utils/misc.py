import os

from contextlib import contextmanager
from typing import Optional


@contextmanager
def as_filename(orig_file: str, new_name: str):
    """
    Copies a file into a new file with the given name

    :param orig_file: Path to the original file
    :type orig_file: str
    :param new_name: Name of the new file
    :type new_name: str
    """
    new_path = None
    try:
        with open(orig_file, "rb") as f:
            content = f.read()

        new_path = os.path.join("/tmp", new_name)
        with open(new_path, "wb") as f:
            f.write(content)

        yield new_path
    finally:
        if new_path is not None:
            os.remove(new_path)


def get_current_git_commit() -> Optional[str]:
    head_file = ".git/HEAD"

    if not os.path.exists(head_file):
        return None

    with open(head_file, "rt") as f:
        content = str(f.read())

    head_ref = content.split(' ')[1].strip()
    head_ref_file = os.path.join(".git", head_ref)

    with open(head_ref_file, "rt") as f:
        return str(f.read()).strip()


def test_1():
    print(get_current_git_commit())


if __name__ == "__main__":
    test_1()
