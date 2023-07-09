from setuptools import setup


setup(
    name='textual_entailment',
    version='0.0.1',    
    description='A python package for performing textual entailment using multiple diverse libraries.',
    url='https://github.com/towhidabsar/textual_entailment',
    author='Towhid Chowdhury',
    author_email='towhidabsar@gmail.com',
    license='MIT',
    packages=['textual_entailment'],
    install_requires=['transformers',
                      'datasets',
                      'torch',
                      'numpy'             
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)