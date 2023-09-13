from setuptools import setup, find_packages

setup(
    name='antifold',
    version='0.1.0',    
    packages=find_packages(),
    description='Inverse folding of antibodies',
    url='https://github.com/Magnushhoie/antifold_web/',
    author='Magnus Haraldson Høie & Alissa Hummer',
    author_email='maghoi@dtu.dk & alissa.hummer@stcatz.ox.ac.uk',
    license='N/A',
    install_requires=['pandas',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
    ],
)
