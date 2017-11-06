#!/bin/bash

cd dist
./install-nest-2.12.0-beskow.sh
cd ..

NEST_INSTALL=$(pwd)/dist/install/nest-simulator-2.12.0

echo "Nest install dir "${NEST_INSTALL}

cd module
./install-module-2.12.0-beskow.sh ${NEST_INSTALL}
