import subprocess

import pytest

from lektor.quickstart import get_default_author
from lektor.quickstart import get_default_author_email


@pytest.fixture
def set_git_config(tmp_path, monkeypatch):
    """Create an isolated git repository and return function which can be used set
    git config values."""
    monkeypatch.chdir(str(tmp_path))
    subprocess.run(["git", "init"], check=True)

    def set_git_config(name, value):
        subprocess.run(["git", "config", name, value], check=True)

    return set_git_config


def test_default_author(os_user):
    assert get_default_author() == "Lektor Test"


def test_default_author_email(set_git_config):
    set_git_config("user.email", "joe@example.org")

    assert get_default_author_email() == "joe@example.org"
