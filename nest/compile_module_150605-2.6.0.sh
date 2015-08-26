
NEST_INSTALL=$(pwd)/dist/install-nest-2.4.2
NEST_SRC=$(pwd)/dist/nest-2.4.2/models
cd module
./compile-module.sh module-150605-2.4.2 nest-2.4.2 $NEST_INSTALL $NEST_SRC
