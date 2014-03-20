#!/bin/sh
#Input: Take module-DATE and nest-version as input
# OBS need to define directory in were nest is installed.

# Directory where nest have been installed
export NEST_INSTALL_DIR="$HOME/opt/NEST/dist/install-$2"
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

#Bootstrap directory, used temoprally. Removed at end of script.
bootstrapDir="$currDir/bootstrap-$1/" 

#Build directory
buildDir="$currDir/build-$1/"

echo "Source dir: $srcDir"
echo ""
echo "Clear previous installation and build directories" 
echo "and init new ones"
echo "Build dir: $buildDir"
echo "Bootstrap dir: $bootstrapDir"
echo "Press [Enter] key to continue..."
read TMP

#Copy source to bootstrap directory
 
rm -r $bootstrapDir
rm -r $buildDir

echo "Copying $srcDir to $bootstrapDir" 
#mkdir $bootstrapDir
cp -r "$srcDir" $bootstrapDir 
echo "Creating build directory"
mkdir $buildDir


echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

#Go into bootstrap dir and run bootstrap
cd $bootstrapDir
./bootstrap.sh 2>&1 | tee ../ml_module-bootstrap.log

#Make new build directory, configure and run make, make install and make installcheck
echo "Entering $buildDir"
cd $buildDir

# To inform about where nest is installed--with-nest=${NEST_INSTALL_DIR}/bin/nest-config
$bootstrapDir"configure" --with-nest=${NEST_INSTALL_DIR}/bin/nest-config 2>&1 | tee ../ml_module-configure.log
make -j $noProcs 2>&1 | tee ../ml_module-make.log
make -j $noProcs install 2>&1 | tee ../ml_module-install.log
#sudo make -j $noProcs installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

# Move out
cd ..

# Move module sli file to path that is in the sli search path
FROM=$NEST_INSTALL_DIR/share/ml_module/sli/ml_module.sli 
TO=$NEST_INSTALL_DIR/share/nest/sli/ml_module.sli
ln -s $FROM $TO

# Create symbolic link to module 
#sudo ln -s $buildDir/ml_module /usr/bin/ml_module

#Display script execution time
echo "It took $DIFF seconds"
