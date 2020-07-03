from setuptools import setup
from setuptools.extension import Extension
import os
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))


# These do not really change the speed of the benchmarks
compiler_directives = {
    'boundscheck': False,
    'language_level': 3,
}

extensions = [
    Extension(
        'pyx.no_fidelities',
        sources=['pyx/no_fidelities.pyx',],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'pyx.sh',
        sources=['pyx/sh.pyx', ],
        include_dirs=[np.get_include()],
    ),
    # Extension(
    #     'resampling_strategies',
    #     sources=['resampling_strategies.py', ],
    #     include_dirs=[np.get_include()],
    # ),
    # Extension(
    #     'greedy_portfolio',
    #     sources=['greedy_portfolio.py', ],
    #     include_dirs=[np.get_include()],
    # )
]

for e in extensions:
    e.cython_directives = compiler_directives



setup(
    name='2020TPAMI',
    ext_modules=extensions,
)
