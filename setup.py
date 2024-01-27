from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="collision-kernels",
    version="1.0",
    description="Computing collision kernels using pychastic",
    url="https://github.com/RadostW/collision-kernels/",
    author="Radost Waszkiewicz and Jan Turczynowicz",
    author_email="radost.waszkiewicz@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/RadostW/collision-kernels",
    },
    license="GNU GPLv3",
    packages=["collision_kernels"],
    install_requires=[
        "numpy",
        "scipy",
        "pychastic",
    ],
    zip_safe=False,
)
