from setuptools import setup

setup(
    name='RnnGen',
    version='1.1',
    packages=['rnngen', 'rnngen/word2vec', 'rnngen/processing', 'rnngen/recurrentnetwork',
              'rnngen/resources', 'rnngen/misc', 'rnngen/predict'],

    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description='Recurrent Neural Network Generative Model with Word2Vec',
    install_requires=[
                    'numpy',
                    'scikit-learn',
                    'matplotlib'
                     ],
    include_package_data=True,
    author='Gabriel Petersson',
    author_email='gabriielpetersson@gmail.com',
    url='https://github.com/gabrielpetersson/rnngen'
)
