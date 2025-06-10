from setuptools import setup, find_packages

setup(
    name="semiq-ml",
    version="0.3.5",
    packages=find_packages(include=["semiq_ml", "semiq_ml.*"]),
    install_requires=[
        # ... your dependencies ...
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
        ],
    },
)
