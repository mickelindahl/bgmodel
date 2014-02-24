#!/bin/sh
#Input: Take source nest directory name (e.g. nest+version) as as parameter
# Example, run by command: 
# sudo ./complile-nest.sh nest-1.9.8718

#Start time watch
START=$(date +%s)

#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo) 

#Build directory
buildDir=build-$1
echo "Build dir: $buildDir"

#Remove old build
echo "Removing previous build directory" 
sudo rm -r build*

#Make new build directory, configure and run make, make install and make installcheck
mkdir $buildDir
echo "Entering $buildDir"
cd $buildDir

# Need to explicilty say where to put nest with --prefix=$HOME/opt/nest
sudo ../$1/configure --prefix=$HOME/opt/nest
sudo make -j $noProcs
sudo make -j $noProcs install  
#sudo make -j $noProcs installcheck

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

#Display script execution time
echo "It took $DIFF seconds"
