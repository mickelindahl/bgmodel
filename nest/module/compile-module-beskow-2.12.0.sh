#!/bin/sh

#Input: Take nest-install-dir

#Examples: 
# ./compile-module-beskow-2.12.0.sh /pdc/vol/nest/2.12.0-py27/


# Directory where nest have been installed
export NEST_INSTALL_DIR="$1"

#enable modules within the batch system
. /opt/modules/default/etc/modules.sh

# compile with the GNU compiler
module swap PrgEnv-cray PrgEnv-gnu
module add nest
module add cmake/2.8.12.2

# Directory where nest have been installed
export NEST_INSTALL_DIR="$1"

MODULE_NAME=module-2.12.0

echo "Nest installation dir: $NEST_INSTALL_DIR"

currDir=$(pwd)
echo $currDir

#Start time watch
START=$(date +%s)

#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo) 

#Source directory
srcDir="$currDir/source/$MODULE_NAME/"

#Build directory
buildDir="$currDir/build/$MODULE_NAME/"

#Install directory
installDir="$currDir/install/$MODULE_NAME/"

#Log directory
logDir="$currDir/log/"
logFileMake="$logDir$MODULE_NAME-make"
logFileInstall="$logDir$MODULE_NAME-install"

echo "Source dir: $srcDir"
echo ""
echo "Clear previous directory and create new one"
echo "Build dir: $buildDir"
echo "Install dir: $installDir"
echo "Source dir: $srcDir"
echo "Log dir: $logDir"
echo "With log files:"
echo "$logFileMake"
echo "$logFileInstall"
echo "Press [Enter] key to continue..."
read TMP

#Copy source to bootstrap directory

if [ -d "$buildDir" ];
then
    echo "Removing old build dir $buildDir"
    rm -r $buildDir
fi

if [ -d "$installDir" ];
then
    echo "Removing old build dir $buildDir"
    rm -r $installDir
fi


echo "Create log dir if it does not exist $logDir"
mkdir -p "$logDir"

echo "Creating build directory if it does not exist $buildDir"
mkdir -p build

echo "Creating install directory if it does not exist $installDir"
mkdir -p $installDir

echo "Copy source to build dir"
cp -r "$srcDir" "$buildDir"

echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

#Go into build dir and run cmake
cd "$buildDir"
cmake -DCMAKE_INSTALL_PREFIX=$installDir -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config
      ../$MODULE_NAME  

# Make and make install
make -j "$noProcs" 2>&1 | tee "$logFileMake"
make -j "$noProcs" install 2>&1 | tee "$logFileInstall"
#sudo make -j $noProcs installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

# Move out
cd ..
