#!/bin/bash

cd dist
./install-nest-2.2.2.sh
cd ..

NEST_INSTALL=$(pwd)/dist/install/nest-2.2.2

cd module
./install-module-2.2.2.sh "$NEST_INSTALL"
