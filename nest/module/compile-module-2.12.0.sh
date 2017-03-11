#!/bin/sh
#Input: Take nest-install-dir

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

#Log directory
logDir="$currDir/log/"
logFileMake="$logDir$1-make"
logFileInstall="$logDir$1-install"

echo "Source dir: $srcDir"
echo ""
echo "Clear previous directory and create new one"
echo "Build dir: $buildDir"
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

echo "Create log dir if it does not exist $logDir"
mkdir -p "$logDir"

echo "Creating build directory if it does not exist $buildDir"
mkdir -p build

echo "Copy source to build dir"
cp -r "$srcDir" "$buildDir"

echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

#Go into build dir and run cmake
cd "$buildDir"
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ../$MODULE_NAME

# Make and make install
make -j "$noProcs" 2>&1 | tee "$logFileMake"
make -j "$noProcs" install 2>&1 | tee "$logFileInstall"
#sudo make -j $noProcs installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

# Move out
cd ..

# Move module sli file to path that is in the sli search path
#FROM=$NEST_INSTALL_DIR/share/ml_module/sli/ml_module.sli 
#TO=$NEST_INSTALL_DIR/share/nest/sli/ml_module.sli
#ln -s $FROM $TO

# Create symbolic link to module 
#sudo ln -s $buildDir/ml_module /usr/bin/ml_module

#Display script execution time
echo "It took $DIFF seconds"
