from setuptools import setup, find_packages

setup(
    name='myoguide-diagnose',
    version='0.1.0',
    author='Jose Verdu-Diaz',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MYO-Guide/MYO-Guide_diagnose',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
