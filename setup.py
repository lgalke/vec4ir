from setuptools import setup

requirements = [
      'boto',
      'bz2file',
      'cycler',
      'decorator',
      'gensim',
      'isodate',
      'matplotlib',
      'networkx',
      'nltk',
      'numpy',
      'pandas',
      'PyYAML',
      'rdflib',
      'requests',
      'scikit-learn',
      'scipy',
      'six',
      'sklearn',
      'smart-open',
      'joblib'
]

setup(name='vec4ir',
      version=0.2,
      description='Neural Word Embeddings for Information Retrieval',
      author="Lukas Galke",
      author_email="lga@informatik.uni-kiel.de",
      install_requires=requirements,
      scripts=['bin/vec4ir-evaluate',
               'bin/vec4ir-run']
      )
