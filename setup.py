from setuptools import setup, find_packages

setup(
    name='NeuralLib',
    version='0.1.0',
    packages=find_packages(include=['NeuralLib', 'NeuralLib.*']),
    install_requires=[
        'torch',
        'huggingface_hub',
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'ecg_peak_detector=NeuralLib.production_models.production_models:ECGPeakDetector'
        ]
    },
    author='Your Name',
    author_email='you@example.com',
    description='Deep Learning Models for Biosignals Processing',
    url='https://github.com/marianaagdias/NeuralLib',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

