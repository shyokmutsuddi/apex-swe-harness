"""Setup configuration for APEX Code Harness."""

from setuptools import setup, find_packages

setup(
    name="apex-code",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "apx=apex_code.cli.main:cli",
        ],
    },
)
