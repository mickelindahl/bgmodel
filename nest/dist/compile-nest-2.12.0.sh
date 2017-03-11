#!/bin/sh

NEST_SRC_FOLDER_NAME=nest-simulator-2.12.0

#Start time watch
START=$(date +%s)

currDir=$(pwd)
echo $currDir
#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo)

#Source directory
srcDir="$currDir/$NEST_SRC_FOLDER_NAME/"

#Build directory
buildDir="$currDir/build-$NEST_SRC_FOLDER_NAME/"

#Build directory
installDir="$currDir/install-$NEST_SRC_FOLDER_NAME/"

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

cmake -DCMAKE_INSTALL_PREFIX:PATH=$insallDir $srcDir
make -j $noProcs 2>&1 | tee $logDir$NEST_SRC_FOLDER_NAME-make
make install 2>&1 | tee $logDir$NEST_SRC_FOLDER_NAME-install
make installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

#Display script execution time
echo "It took $DIFF seconds"
