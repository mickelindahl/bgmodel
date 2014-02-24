#!/bin/sh
#Take nest+version as as parameter. OBS nest will be installed 
#install-nest+version in current directory. For nest to work
#LD_LIBRARY_PATH in bash and eclipse has to point to this directory.
# Also this directory have to be provided when compling the 
# module.

#Start time watch
START=$(date +%s)

currDir=$(pwd)
echo $currDir
#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo) 

#Build directory
buildDir="$currDir/build-$1"

#Build directory
installDir="$currDir/install-$1"

#Remove old build
echo "Clear previous installation and build directories" 
echo "Build dir: $buildDir"
echo "Install dir: $installDir"
echo "Press [Enter] key to continue..."
read TMP
rm -r $buildDir
rm -r $installDir

#Make new build directory, configure and run make, make install and make installcheck
echo "Create new build and install dir directory: "
echo $buildDir 
echo $installDir 
mkdir $buildDir
mkdir $installDir
echo "Entering $buildDir"
cd $buildDir 

echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

# Need to explicilty say where to put nest with --prefix=$HOME/opt/nest
../$1/configure --with-mpi --prefix=$installDir 2>&1 | tee "../mymodule-configure-$2.log"
make -j $noProcs2 >&1 | tee "../mymodule-make-$2.log"
sudo make -j $noProcs install "../mymodule-install-$2.log"
#sudo make -j $noProcs installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

#Display script execution time
echo "It took $DIFF seconds"
