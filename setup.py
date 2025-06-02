from setuptools import setup, find_packages

setup(
    name="semiq-ml",
    version="0.1.0-release",
    packages=find_packages(include=["semiq_ml", "semiq_ml.*"]),
    install_requires=[
        # ... your dependencies ...
    ],
)
