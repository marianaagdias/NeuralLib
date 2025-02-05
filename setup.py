from setuptools import setup, find_packages
import shutil
import os

# Ensure the right config file is used for the pip package
shutil.copy("NeuralLib/utils/config_pip.py", "NeuralLib/utils/config.py")


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name='NeuralLib',
    version='0.1.0',
    packages=find_packages(include=['NeuralLib', 'NeuralLib.*']),
    install_requires=parse_requirements("requirements.txt"),  # Loads dependencies dynamically from requirements.txt
    include_package_data=True,  # Ensures config.py is included
    package_data={
        "NeuralLib": ["utils/config.py"],  # Only add the pip config
    },
    author='Mariana Dias',
    author_email='marianaagdias97@gmail.com',
    description='Deep Learning Models for Biosignals Processing',
    url='https://github.com/marianaagdias/NeuralLib',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)

# Cleanup after packaging
if os.path.exists("NeuralLib/utils/config.py"):
    os.remove("NeuralLib/utils/config.py")
