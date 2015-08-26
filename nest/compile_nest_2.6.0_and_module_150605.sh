cd dist
./compile-nest-mpi.sh nest-2.6.0
cd ..

NEST_INSTALL=$(pwd)/dist/install-nest-2.6.0
NEST_SRC=$(pwd)/dist/nest-2.6.0/models
cd module
./compile-module.sh module-150605-2.6.0 nest-2.6.0 $NEST_INSTALL $NEST_SRC
