#!/bin/sh
#Input: Take module-DATE and nest-version as input
# OBS need to define directory in were nest is installed.

# Directory where nest have been installed
export NEST_INSTALL_DIR="$HOME/tools/NEST/dist/install-$2"
echo "Nest instalation dir: $NEST_INSTALL_DIR"


#Start time watch
START=$(date +%s)

#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo) 

#Source directory
srcDir=$1

#Bootstrap directory, used temoprally. Removed at end of script.
bootstrapDir=bootstrap-$1 

#Build directory
buildDir=build-$1

echo "Source dir: $srcDir"
echo "Build dir: $buildDir"
echo "Bootstrap dir: $bootstrapDir"

#Copy source to bootstrap directory
rm -r $bootstrapDir
cp -r $srcDir $bootstrapDir

echo "Press [Enter] key to continue..."
read TMP

#Go into bootstrap dir and run bootstrap
cd $bootstrapDir
./bootstrap.sh 2>&1 | tee ../mymodule-bootstrap.log

#Move out
cd ..

#Remove old build
echo "Removing previous build directory" 
rm -r build*

#Make new build directory, configure and run make, make install and make installcheck
mkdir $buildDir
echo "Entering $buildDir"
cd $buildDir

# To inform about where nest is installed--with-nest=${NEST_INSTALL_DIR}/bin/nest-config
../$bootstrapDir/configure --with-nest=${NEST_INSTALL_DIR}/bin/nest-config 2>&1 | tee ../mymodule-configure.log
make -j $noProcs 2>&1 | tee ../mymodule-make.log
make -j $noProcs install 2>&1 | tee ../mymodule-install.log
#sudo make -j $noProcs installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

# Move out
cd ..


#Remove bootstrap directory
#echo "Removing bootstrap directory" 
#rm -r $bootstrapDir

# Create symbolic link to module 
#sudo ln -s $buildDir/ml_module /usr/bin/ml_module

#Display script execution time
echo "It took $DIFF seconds"
