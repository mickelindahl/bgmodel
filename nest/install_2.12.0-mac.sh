#!/bin/bash

cd dist
./install-nest-2.12.0-mac.sh
cd ..

NEST_INSTALL=$(pwd)/dist/install/nest-simulator-2.12.0

cd module
./install-module-2.12.0-mac.sh "$NEST_INSTALL"
