#!/usr/bin/env sh

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

if [[ "$SINGULARITY_VERSION" == "3.5" ]]; then
    export VERSION=3.5.3
[[ "$SINGULARITY_VERSION" == "3.6" ]]; then
    export VERSION=3.6.4
[[ "$SINGULARITY_VERSION" == "3.7" ]]; then
    export VERSION=3.7.3
else
    echo "Skip installing Singularity"
fi

wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz && \
tar -xzf singularity-${VERSION}.tar.gz && \
cd ./singularity && \
./mconfig && \
make -C builddir && \
sudo make -C builddir install

cd ..
pip install .[singulaity]
