"""Setup configuration for APEX SWE Harness."""

from setuptools import setup, find_packages

setup(
    name="apex-swe-harness",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "apex-runner=apex_harness.cli:main",
        ],
    },
)
