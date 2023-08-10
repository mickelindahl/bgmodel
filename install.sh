#!/bin/bash

eval $(cat .env)

NEST_VERSION=3.5

#Start time watch
START=$(date +%s)

./scripts/install-nest-conda.sh -y "$NEST_VERSION" "$PATH_CONDA_SETUP"
cp ./nest/dist/conda-activate.sh .


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
echo -e "   \033[36m$(cat ./conda-activate.sh)\033[0m"
echo ""
echo -e "2. Using provided script:"
echo -e "   \033[36msource ./conda-activate.sh\033[0m created in root"
echo ""
echo "---------------------------------------------"