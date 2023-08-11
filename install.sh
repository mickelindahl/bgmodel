#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

eval $(cat .env)

NEST_VERSION=3.5

#Start time watch
START=$(date +%s)

./scripts/install-nest-conda.sh -y --ignore-conda-cache "$NEST_VERSION" "$PATH_CONDA_SETUP"
mv ./scripts/conda-activate.sh .

MODULE_SOURCE_DIR=$SCRIPT_DIR/nest/module/source/ml_module-$NEST_VERSION
NEST_INSTALL_DIR=$SCRIPT_DIR/nest/dist/install/nest-simulator-$NEST_VERSION

./scripts/install-module-conda.sh -y "$MODULE_SOURCE_DIR" "$NEST_INSTALL_DIR" "$PATH_CONDA_SETUP"

#Stop time watch
END=$(date +%s)
DIFF_TOTAL=$(("$END" - "$START"))

echo ""
echo ""
echo -e "\033[32mDone!\033[0m"
echo "---------------------------------------------"
echo -e "Installation took \033[33m$DIFF_TOTAL\033[0m seconds"
echo "---------------------------------------------"
echo ""
echo -e "\033[1mActivation:\033[0m"
echo -e "1. Using \033[34mconda\033[0m command:"
echo -e "   \033[36mconda activate $NEST_INSTALL_DIR\033[0m"
echo ""
echo -e "2. Using provided script:"
echo -e "   \033[36msource ./conda-activate.sh\033[0m created in root"
echo ""
echo "---------------------------------------------"