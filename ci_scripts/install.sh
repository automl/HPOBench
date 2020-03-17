#!/usr/bin/env sh

install_packages=""

if [[ "$RUN_TESTS" == "true" ]]; then
    echo "Install tools for testing"
    install_packages="${install_packages}pytest,"
    pip install codecov
else
    echo "Skip installing tools for testing"
fi

if [[ "$RUN_CODESTYLE" == "true" ]]; then
    echo "Install tools for codestyle checking"
    install_packages="${install_packages}codestyle,"
else
    echo "Skip installing tools for codestyle checking"
fi

if [[ "$USE_SINGULARITY" == "true" ]]; then
    echo "Install Singularity"
    gimme force 1.14
    eval "$(gimme 1.14)"

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

    export VERSION=3.5.2 && # adjust this as necessary \
      wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz && \
      tar -xzf singularity-${VERSION}.tar.gz && \
      cd singularity

    ./mconfig && \
      make -C builddir && \
      sudo make -C builddir install

    cd ..
    install_packages="${install_packages}singularity,"
else
    echo "Skip installing Singularity"
fi

install_packages="${install_packages}xgboost"
echo "Install HPOlib3 with options: ${install_packages}"
pip install .["${install_packages}"]
