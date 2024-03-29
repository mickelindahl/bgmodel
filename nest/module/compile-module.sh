#!/bin/sh
#Input: Take module-DATE, nest-version, nest-install-dir and nest-source-dir/models as input
#Examples: 
# compile-module-milner module-130701 nest-2.2.2 /pdc/vol/nest/2.2.2/ /afs/nada.kth.se/home/w/u1yxbcfw/opt/NEST/dist/nest-2.2.2/models
#./compile-module.sh module-130701 nest-2.4.2 /home/mikael/opt/NEST/dist/install-nest-2.4.2/ /home/mikael/opt/NEST/dist/nest-2.4.2/models

# Directory where nest have been installed
export NEST_INSTALL_DIR="$3"
export NEST_MODELS_DIR="$4" #directory for of soruce code for models. Needed in Makefile.am

#export NEST_INSTALL_DIR="$HOME/opt/NEST/dist/install-$2"
#export NEST_INSTALL_DIR="/home/mikael/opt/NEST/dist/install-$2"
echo "Nest instalation dir: $NEST_INSTALL_DIR"

currDir=$(pwd)
echo $currDir

#Start time watch
START=$(date +%s)

#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo) 

#Source directory
srcDir="$currDir/$1/"

#Build directory
buildDir="$currDir/build-$1-$2/"

#Log directory
logDir="$currDir/log/"

echo "Source dir: $srcDir"
echo ""
echo "Clear previous directory and create new one"
echo "Build dir: $buildDir"
echo "Source dir: $srcDir"
echo "Log dir: $logDir"
echo "Press [Enter] key to continue..."
read TMP

#Copy source to bootstrap directory
 
if [ -d "$buildDir" ]; then rm -r $buildDir
fi
if [ ! -d "$logDir" ]; then mkdir $logDir
fi


echo "Creating build directory"
mkdir $buildDir

echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

#Go into build dir and run cmake
cd $buildDir
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ../$1

# Make and make install
make -j $noProcs 2>&1 | tee $logDir$1-$2-make
make -j $noProcs install 2>&1 | tee $logDir$1-$2-install
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
