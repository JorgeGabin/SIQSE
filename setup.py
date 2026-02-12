from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="siqse",
    version="0.1.0",
    author="Jorge Gabín",
    author_email="jorge.gabin@udc.es",
    description="Simulation-based Interactive Query Suggestion Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JorgeGabin/SIQSE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Query Suggestion Selection",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
)
