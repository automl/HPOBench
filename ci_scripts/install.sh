#!/usr/bin/env sh

install_packages=""

if [[ "$RUN_TESTS" == "true" ]]; then
    echo "Install tools for testing"
    install_packages="${install_packages}xgboost,pytest,test_paramnet,test_tabular_datamanager,"
    pip install codecov

    # The param net benchmark does not work with a scikit-learn version != 0.23.2. (See notes in the benchmark)
    # To make sure that no newer version is installed, we install it before the other requirements.
    # Since we are not using a "--upgrade" option later on, pip skips to install another scikit-learn version.
    echo "Install the right scikit-learn function for the param net tests."
    pip install --upgrade scikit-learn==0.23.2
else
    echo "Skip installing tools for testing"
fi

if [[ "$RUN_CODESTYLE" == "true" ]]; then
    echo "Install tools for codestyle checking"
    install_packages="${install_packages}codestyle,"
else
    echo "Skip installing tools for codestyle checking"
fi

if [[ "$RUN_CONTAINER_EXAMPLES" == "true" ]]; then
    echo "Install packages for container examples"
    echo "Install swig"
    sudo apt-get update && sudo apt-get install -y build-essential swig
else
    echo "Skip installing packages for container examples"
fi

if [[ "$RUN_LOCAL_EXAMPLES" == "true" ]]; then
    echo "Install packages for local examples"
    echo "Install swig"
    sudo apt-get update && sudo apt-get install -y build-essential swig
    install_packages="${install_packages}xgboost,"
else
    echo "Skip installing packages for local examples"
fi

if [[ "$USE_SINGULARITY" == "true" ]]; then
    echo "Install Singularity"

    sudo apt-get update && sudo apt-get install -y \
      build-essential \
      libssl-dev \
      uuid-dev \
      libgpgme11-dev \
      squashfs-tools \
      libseccomp-dev \
      wget \
      pkg-config \
      git \
      cryptsetup

    export VERSION=3.5.3 && # adjust this as necessary \
      wget https://github.com/sylabs/singularity/archive/refs/tags/v${VERSION}.tar.gz && \
      tar -xzf v${VERSION}.tar.gz && \
      cd singularity-${VERSION}

    ./mconfig && \
      make -C builddir && \
      sudo make -C builddir install

    cd ..
    install_packages="${install_packages}placeholder,"
else
    echo "Skip installing Singularity"
fi

# remove the trailing comma
install_packages="$(echo ${install_packages} | sed 's/,*\r*$//')"
echo "Install HPOBench with options: ${install_packages}"
pip install .["${install_packages}"]
