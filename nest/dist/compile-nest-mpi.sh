#!/bin/sh
#Take nest+version as as parameter. OBS nest will be installed 
#install-nest+version in current directory. For nest to work
#LD_LIBRARY_PATH in bash and eclipse has to point to this directory.
# Also this directory have to be provided when compling the 
# module.

# OBS Strange error, sometimes need to delete install folder by hand to make it work

#Start time watch
START=$(date +%s)

currDir=$(pwd)
echo $currDir
#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo) 

#Source directory
srcDir="$currDir/$1/"

#Build directory
buildDir="$currDir/build-$1/"

#Build directory
installDir="$currDir/install-$1/"

#Log directory
logDir="$currDir/log/"

#Remove old build
echo "Clear previous installation and build directories" 
echo "Build dir: $buildDir"
echo "Install dir: $installDir"
echo "Log dir: $logDir"
echo "Press [Enter] key to continue..."
read TMP

if [ -d "$buildDir" ]; 
then
echo "Removing $buildDir"
rm -r $buildDir 
fi

if [ -d "$installDir" ]; 
then 
echo "Removing $installDir"
rm -r $installDir 
fi

if [ ! -d "$logDir" ]; then mkdir $logDir 
fi

echo "Press [Enter] key to continue..."
read TMP

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
$srcDir"configure" --with-mpi --prefix=$installDir 2>&1 | tee $logDir$1-configure
make -j $noProcs 2>&1 | tee $logDir$1-make
make -j $noProcs install 2>&1 | tee $logDir$1-install
#make -j $noProcs installcheck 2>&1 | tee $logDir$1-installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

#Display script execution time
echo "It took $DIFF seconds"
