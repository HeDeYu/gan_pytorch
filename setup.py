#!/usr/bin/env python

"""The setup script."""

import io
import os
import sys

# Python supported version checks. Keep right after stdlib imports to ensure we
# get a sensible error for older Python versions
if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")


from setuptools import find_packages, setup

import versioneer


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fh:
        return fh.read()


readme = read("README.rst")
changelog = read("CHANGELOG.rst")

install_requires = [
    # eg: "numpy==1.11.1", "six>=1.7",
]

extras_require = {
    "dev": [
        "black==22.3.0",
        "isort==5.7.0",
        "flake8==3.8.4",
        "mypy==0.800",
        "pre-commit~=2.10.0",
        "pytest==6.2.2",
        "pytest-cov==2.11.1",
        "tox~=3.21.0",
        "gitchangelog==3.0.4",
        "invoke==1.5.0",
    ]
}


def setup_package():
    metadata = dict(
        author="Deyu He",
        author_email="18565286660@163.com",
        python_requires=">=3.6",
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        description="An example package. Generated with cookiecutter-rrpylibrary.",
        install_requires=install_requires,
        extras_require=extras_require,
        long_description=readme + "\n\n" + changelog,
        include_package_data=True,
        keywords="gan_pytorch",
        name="gan_pytorch",
        url="https://github.com/HeDeYu/gan_pytorch",
        version=versioneer.get_version(),
        package_dir={"": "src"},
        zip_safe=False,
        cmdclass=versioneer.get_cmdclass(),
        packages=find_packages("src"),
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
