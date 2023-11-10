#!/usr/bin/env sh

install_packages=""

if [[ "$RUN_TESTS" == "true" ]]; then
    echo "Install tools for testing"
    install_packages="${install_packages}pytest,test_tabular_datamanager,"
    pip install codecov

    PYVERSION=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]*\).*/\1\2/')
    if [[ "${PYVERSION}" != "310" ]]; then
      # The param net benchmark does not work with a scikit-learn version != 0.23.2. (See notes in the benchmark)
      # To make sure that no newer version is installed, we install it before the other requirements.
      # Since we are not using a "--upgrade" option later on, pip skips to install another scikit-learn version.
      echo "Install the right scikit-learn function for the param net tests."
      pip install --upgrade scikit-learn==0.23.2
      install_packages="${install_packages}xgboost,test_paramnet,"
    else
      echo "Skip installing the extra paramnet tests."
      # For 3.10, we need a different pandas version - this comes as a requirement for the old xgboost benchmark.
      # building pandas<=1.5.0 does not work with 3.10 anymore. -> install a different version.
      install_packages="${install_packages}xgboost_310,"
    fi

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

    PYVERSION=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]*\).*/\1\2/')
    if [[ "${PYVERSION}" != "310" ]]; then
      # For 3.10, we need a different pandas version - this comes as a requirement for the old xgboost benchmark.
      # building pandas<=1.5.0 does not work with 3.10 anymore. -> install a different version.
      install_packages="${install_packages}xgboost,"
    else
      install_packages="${install_packages}xgboost_310,"
    fi

else
    echo "Skip installing packages for local examples"
fi

# We add a placeholder / No-OP operator. When running the container examples, we don't install any
# additional packages. That causes an error, since `pip install .[]` does not work.
install_packages="${install_packages}NOP,"

# remove the trailing comma
install_packages="$(echo ${install_packages} | sed 's/,*\r*$//')"
echo "Install HPOBench with options: ${install_packages}"
pip install .["${install_packages}"]