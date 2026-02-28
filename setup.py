from setuptools import find_packages, setup

setup(
    name="grid-predictive-maintenance",
    version="0.1.0",
    description="Predictive maintenance system for utility analytics teams",
    packages=find_packages(include=["src", "src.*", "api", "monitoring"]),
    include_package_data=True,
    python_requires=">=3.10",
)
