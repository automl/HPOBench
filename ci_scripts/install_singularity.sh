#!/usr/bin/env sh

echo "Inside Singularity Installation Script"

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

if [[ "$SINGULARITY_VERSION" == "3.5" ]]; then
    export VERSION=3.5.3
elif [[ "$SINGULARITY_VERSION" == "3.6" ]]; then
    export VERSION=3.6.4
elif [[ "$SINGULARITY_VERSION" == "3.7" ]]; then
    export VERSION=3.7.3
elif [[ "$SINGULARITY_VERSION" == "3.8" ]]; then
    export VERSION=3.8.4
elif [[ "$SINGULARITY_VERSION" == "3.9" ]]; then
    export VERSION=3.9.3
elif [[ "$SINGULARITY_VERSION" == "3.10" ]]; then
    export VERSION=3.10.1
else
    echo "Skip installing Singularity"
fi

wget https://github.com/singularityware/singularity/releases/download/v"${VERSION}"/singularity-"${VERSION}".tar.gz && \
tar -xzf singularity-"${VERSION}".tar.gz && \
cd singularity-"${VERSION}" && \
./mconfig && \
make -C builddir && \
sudo make -C builddir install

cd ..
pip install .
