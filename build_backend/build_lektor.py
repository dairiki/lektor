""" PEP-517 backend to build Lektor distribution.

This is a custom `in-tree build backend`_.  It is a thin-wrapper
around `setuptools.build_meta` which ensures that the static files
for the admin UI are built before an sdist or wheel is built.

.. _in-tree build backend: https://www.python.org/dev/peps/pep-0517/#in-tree-build-backends

"""
import subprocess

import setuptools.build_meta
from setuptools.build_meta import get_requires_for_build_sdist  # noqa: F401
from setuptools.build_meta import get_requires_for_build_wheel  # noqa: F401
from setuptools.build_meta import prepare_metadata_for_build_wheel  # noqa: F401


class AdminBuilder:
    def __init__(self, path="lektor/admin"):
        self.path = path

    def run(self, cmd):
        print(f"Running {' '.join(cmd)}")
        subprocess.run(cmd, cwd=self.path, check=True)

    def __call__(self):
        self.run(["npm", "install"])
        self.run(["npm", "run", "webpack"])


build_admin = AdminBuilder()


def build_wheel(*args, **kwargs):
    build_admin()
    return setuptools.build_meta.build_wheel(*args, **kwargs)


def build_sdist(*args, **kwargs):
    build_admin()
    return setuptools.build_meta.build_sdist(*args, **kwargs)
