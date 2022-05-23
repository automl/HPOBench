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

if [[ "$SINGULARITY_VERSION" == "3.7" ]]; then
    export VERSION=3.7.3
    export FILENAME=singularity-"${VERSION}"
    export EXTRACTED_FILENAME=singularity

elif [[ "$SINGULARITY_VERSION" == "3.8" ]]; then
    export VERSION=3.8.4
    export FILENAME=singularity-ce-"${VERSION}"
    export EXTRACTED_FILENAME=singularity-ce-"${VERSION}"

elif [[ "$SINGULARITY_VERSION" == "3.9" ]]; then
    export VERSION=3.9.3
    export FILENAME=singularity-ce-"${VERSION}"
    export EXTRACTED_FILENAME=singularity-ce-"${VERSION}"

elif [[ "$SINGULARITY_VERSION" == "3.10" ]]; then
    export VERSION=3.10.0
    export FILENAME=singularity-ce-"${VERSION}"
    export EXTRACTED_FILENAME=singularity-ce-"${VERSION}"

else
    echo "Skip installing Singularity"
fi

wget https://github.com/sylabs/singularity/releases/download/v"${VERSION}"/"${FILENAME}".tar.gz && \
tar -xzf "${FILENAME}".tar.gz && \
cd "${EXTRACTED_FILENAME}" && \
./mconfig && \
make -C builddir && \
sudo make -C builddir install

cd ..
pip install .
