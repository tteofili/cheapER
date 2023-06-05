import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cheapER", # Replace with your own username
    version="0.0.1",
    author="Tommaso Teofili",
    author_email="tommaso.teofili@gmail.com",
    description="Cheap Entity Resolution pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/tteofili/cheapER.git',
    packages=['cheaper', 'cheaper.data', 'cheaper.emt', 'cheaper.similarity'],
    install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'scikit-learn',
          'tqdm',
          'transformers',
          'torch',
          'datasets',
          'matplotlib',
          'textdistance',
          'strsimpy',
          'nltk',
          'datasketch',
          'tensorboardX',
          'wandb'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
