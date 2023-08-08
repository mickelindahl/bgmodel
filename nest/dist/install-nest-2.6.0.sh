#!/bin/sh

NEST_VERSION=2.6.0
NEST_FOLDER_NAME=nest-2.6.0
NEST_TAR=nest-$NEST_VERSION.tar.gz

echo ""

if [ -d "source/$NEST_FOLDER_NAME" ];
then

    echo "Source files already downloaded"

else

    mkdir -p source

    echo "Entering source folder"
    cd source
    URL="https://github.com/nest/nest-simulator/releases/download/v$NEST_VERSION/$NEST_TAR"
    echo "Downloading nest from $URL"
    wget $URL

    echo "Unpacking "$NEST_TAR" to source folder"
    tar -zxvf "$NEST_TAR"
    cd ..
fi


echo ""
echo "Proceeding with installation"
echo ""
#Start time watch
START=$(date +%s)


currDir=$(pwd)

#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo)

#Source directory
srcDir="$currDir/source/$NEST_FOLDER_NAME/"

#Build directory
buildDir="$currDir/build/$NEST_FOLDER_NAME/"

#Build directory
installDir="$currDir/install/$NEST_FOLDER_NAME/"

#Log directory
logDir="$currDir/log/"

#echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
#
## Directory where nest have been installed
#export NEST_INSTALL_DIR="$1"
#export LD_LIBRARY_PATH="$installDir/lib/nest"
#
#echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

#Remove old build
echo "Clear previous installation and build directories"
echo "Source dir: $srcDir"
echo "Build dir: $buildDir"
echo "Install dir: $installDir"
echo "Log dir: $logDir"
echo "Press [Enter] key to continue..."
read TMP

if [ -d "$buildDir" ];
then
    echo "Removing $buildDir"
    rm -r "$buildDir"

else
    echo "No previous build dir to remove"
fi

echo "Creating build dir $buildDir"
mkdir -p "$buildDir"

if [ -d "$installDir" ];
then
    echo "Removing $installDir"
    rm -r "$installDir"
else
    echo "Not previous install dir to remove"
fi

echo "Creating install dir $installDir"
mkdir -p "$installDir"

echo "Create log dir if it does not exist $logDir"
mkdir -p "$logDir"

echo "Enter build dir $buildDir"
cd "$buildDir"

echo "Press [Enter] key to continue..."
read TMP

# Need to explicilty say where to put nest with --prefix=$HOME/opt/nest
$srcDir"configure" --prefix=$installDir 2>&1 | tee "$logDir$NEST_FOLDER_NAME-configure"
make -j $noProcs 2>&1 | tee "$logDir$NEST_FOLDER_NAME-make"
make -j $noProcs install 2>&1 | tee "$logDir$NEST_FOLDER_NAME-install"
#make -j $noProcs installcheck 2>&1 | tee $logDir$1-installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

#Display script execution time
echo "It took $DIFF seconds"
