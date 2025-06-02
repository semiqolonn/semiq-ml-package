from setuptools import setup, find_packages

setup(
    name="semiq-ml",
    version="0.1.1a",
    packages=find_packages(include=["semiq_ml", "semiq_ml.*"]),
    install_requires=[
        # ... your dependencies ...
    ],
)
