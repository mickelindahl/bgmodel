cd dist
./compile-nest-mpi.sh nest-2.2.2
cd ..

NEST_INSTALL=$(pwd)/dist/install-nest-2.2.2
NEST_SRC=$(pwd)/dist/nest-2.2.2/models
cd module
./compile-module.sh module-130701 nest-2.2.2 $NEST_INSTALL $NEST_SRC
