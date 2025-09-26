from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py_target_id",
    version="0.1.0",
    author="Jeffrey Granja",
    author_email="jgranja@cartography.bio",
    description="Target ID analysis tools for cartography.bio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/target_id",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0,<2.2.0",  # Avoid pandas 2.2+ NumPy 2.0 issues
        "numpy>=1.20.0,<2.0",    # Stay on NumPy 1.x
        "scipy>=1.7.0,<1.14.0",  # Compatible scipy range
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
)