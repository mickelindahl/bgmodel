#!/bin/sh
#Input: Take nest-install-dir

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Directory where nest have been installed
export NEST_INSTALL_DIR="$1"
export LD_LIBRARY_PATH="$NEST_INSTALL_DIR/lib/nest"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

MODULE_NAME=module-2.2.2

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

#Bootstrap directory, used temoprally. Removed at end of script.
bootstrapDir="$currDir/bootstrap/$MODULE_NAME/"

#Log directory
logDir="$currDir/log/"
logFileMake="$logDir$MODULE_NAME-make"
logFileInstall="$logDir$MODULE_NAME-install"
logFileBootstrap="$logDir$MODULE_NAME-bootstrap"
logFileConfigure="$logDir$MODULE_NAME-configure"

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

if [ -d "$bootstrapDir" ];
then
    echo "Removing old bootstrap dir $bootstrapDir"
    rm -r $bootstrapDir
fi

echo "Create log dir if it does not exist $logDir"
mkdir -p "$logDir"

echo "Creating build directory if it does not exist $buildDir"
mkdir -p $buildDir

echo "Creating bootstrap directory if it does not exist $bootstrapDir"
mkdir -p bootstrap

echo "Copying $srcDir to $bootstrapDir"
cp -r "$srcDir" $bootstrapDir

echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

echo "Go into bootstrap dir and run bootstrap"
cd $bootstrapDir
./bootstrap.sh 2>&1 | tee $logFileBootstrap

#Make new build directory, configure and run make, make install and make installcheck
echo "Entering $buildDir"
cd $buildDir

$bootstrapDir"configure" --with-nest="${NEST_INSTALL_DIR}/bin/nest-config" --prefix=${NEST_INSTALL_DIR}/ 2>&1 | tee $logFileConfigure
#$bootstrapDir"configure" --with-nest="${NEST_INSTALL_DIR}/bin/nest-config" | tee $logDir$1-$2-configure
make -j $noProcs 2>&1 | tee $logFileMake
make -j $noProcs install 2>&1 | tee $logFileInstall
#sudo make -j $noProcs installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

echo "Copy ${NEST_INSTALL_DIR}/share/ml_module/sli/ml_module.sli to ${NEST_INSTALL_DIR}/share/nest/sli/ml_module.sli"
echo "to make it visible for nest.Install"
cp ${NEST_INSTALL_DIR}/share/ml_module/sli/ml_module.sli ${NEST_INSTALL_DIR}/share/nest/sli/ml_module.sli

#Display script execution time
echo ""
echo "It took $DIFF seconds"
echo "Module installed to ${NEST_INSTALL_DIR}"

echo ""
echo "Make sure LD_LIBRARY_PATH includes ${NEST_INSTALL_DIR}/lib/nest/"
echo "To be sure run in python"
echo ">> import os"
echo ">> print os.environ['LD_LIBRARY_PATH']"
echo "and make user the path is included"
echo "Otherwise set it with"
echo ">> os.environ['LD_LIBRARY_PATH']=${NEST_INSTALL_DIR}/lib/nest/"

