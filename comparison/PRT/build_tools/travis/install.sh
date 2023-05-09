#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone scikit-learn/scikit-learn repository in to a local repository.
# We use a cached directory with three scikit-learn repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Install miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    MINICONDA_PATH=/home/travis/miniconda
    chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
    export PATH=$MINICONDA_PATH/bin:$PATH
    conda update --yes conda

    TO_INSTALL="python=$PYTHON_VERSION pip pytest pytest-cov \
                numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
                cython=$CYTHON_VERSION"

    if [[ "$INSTALL_MKL" == "true" ]]; then
        TO_INSTALL="$TO_INSTALL mkl"
    else
        TO_INSTALL="$TO_INSTALL nomkl"
    fi

    if [[ -n "$PANDAS_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pandas=$PANDAS_VERSION"
    fi

    if [[ -n "$PYAMG_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pyamg=$PYAMG_VERSION"
    fi

    if [[ -n "$PILLOW_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pillow=$PILLOW_VERSION"
    fi

    conda create -n testenv --yes $TO_INSTALL
    source activate testenv

    # for python 3.4, conda does not have recent pytest packages
    if [[ "$PYTHON_VERSION" == "3.4" ]]; then
        pip install pytest==3.5
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to use the numpy installed through apt-get
    # install.
    deactivate
    # Create a new virtualenv using system site packages for python, numpy
    # and scipy
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install pytest pytest-cov cython==$CYTHON_VERSION

elif [[ "$DISTRIB" == "scipy-dev-wheels" ]]; then
    # Set up our own virtualenv environment to avoid travis' numpy.
    # This venv points to the python interpreter of the travis build
    # matrix.
    virtualenv --python=python ~/testvenv
    source ~/testvenv/bin/activate
    pip install --upgrade pip setuptools

    echo "Installing numpy and scipy master wheels"
    dev_url=https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com
    pip install --pre --upgrade --timeout=60 -f $dev_url numpy scipy pandas cython
    pip install pytest pytest-cov
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage codecov
fi

if [[ "$TEST_DOCSTRINGS" == "true" ]]; then
    pip install sphinx numpydoc  # numpydoc requires sphinx
fi

if [[ "$SKIP_TESTS" == "true" && "$CHECK_PYTEST_SOFT_DEPENDENCY" != "true" ]]; then
    echo "No need to build scikit-learn"
else
    # Build scikit-learn in the install.sh script to collapse the verbose
    # build output in the travis output when it succeeds.
    python --version
    python -c "import numpy; print('numpy %s' % numpy.__version__)"
    python -c "import scipy; print('scipy %s' % scipy.__version__)"
    python -c "\
try:
    import pandas
    print('pandas %s' % pandas.__version__)
except ImportError:
    pass
"
    python setup.py develop
    ccache --show-stats
    # Useful for debugging how ccache is used
    # cat $CCACHE_LOGFILE
fi

if [[ "$RUN_FLAKE8" == "true" ]]; then
    conda install flake8 -y
fi
