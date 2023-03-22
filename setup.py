import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = ["mkdocs==1.3.0", "mkdocstrings==0.18.1"]
style_packages = ["black==23.1.0", "flake8==6.0.0", "isort==5.12.0"]
test_packages = ["pytest==7.2.2", "pytest-cov"] #, "great-expectations==0.15.15"]


setuptools.setup(
    name="micrograd",
    version="0.1.0",
    author="Andrej Karpathy",
    author_email="andrej.karpathy@gmail.com",
    description="A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karpathy/micrograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages + style_packages + test_packages + ["pre-commit==2.19.0"],
        "docs": docs_packages,
        "test": test_packages,
    },
)
