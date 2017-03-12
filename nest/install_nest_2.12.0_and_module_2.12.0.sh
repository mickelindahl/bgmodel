#!/bin/bash

cd dist
./install-nest-2.12.0.sh
cd ..

NEST_INSTALL=$(pwd)/dist/install/nest-simulator-2.12.0

cd module
./install-module-2.12.0.sh "$NEST_INSTALL"
