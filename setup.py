import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="restricted_boltzmann",
    version="0.0.4",
    author="J. A. Moreno-Guerra",
    author_email="jzs.gm27@gmail.com",
    description="Testing installation of Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzsmoreno/restricted_boltzmann",
    project_urls={"Bug Tracker": "https://github.com/jzsmoreno/restricted_boltzmann"},
    license="MIT",
    packages=["restricted_boltzmann"],
    install_requires=["matplotlib", "tensorflow"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
