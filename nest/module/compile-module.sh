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

#Bootstrap directory, used temoprally. Removed at end of script.
bootstrapDir="$currDir/bootstrap-$1-$2/" 

#Build directory
buildDir="$currDir/build-$1-$2/"

#Build directory
installDir="$currDir/install-$1-$2/"

#Log directory
logDir="$currDir/log/"

echo "Source dir: $srcDir"
echo ""
echo "Clear previous installation and build directories and init new ones"
echo "Build dir: $buildDir"
echo "Bootstrap dir: $bootstrapDir"
echo "Install dir: $installDir"
echo "Log dir: $logDir"
echo "Press [Enter] key to continue..."
read TMP

#Copy source to bootstrap directory
 
if [ -d "$bootstrapDir" ]; then rm -r $bootstrapDir 
fi
if [ -d "$buildDir" ]; then rm -r $buildDir 
fi
if [ -d "$installDir" ]; then rm -r $installDir 
fi
if [ ! -d "$logDir" ]; then mkdir $logDir 
fi


echo "Copying $srcDir to $bootstrapDir" 
#mkdir $bootstrapDir
cp -r "$srcDir" $bootstrapDir 
echo "Creating build directory"
mkdir $buildDir
mkdir $installDir

echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

#Go into bootstrap dir and run bootstrap
cd $bootstrapDir
./bootstrap.sh 2>&1 | tee $logDir$1-$2-bootstrap

#Make new build directory, configure and run make, make install and make installcheck
echo "Entering $buildDir"
cd $buildDir



# To inform about where nest is installed --with-nest=${NEST_INSTALL_DIR}/bin/nest-config and with --prefix inform where to put module installation files


$bootstrapDir"configure" --with-nest="${NEST_INSTALL_DIR}/bin/nest-config" --prefix=$installDir/ 2>&1 | tee $logDir$1-$2-configure
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
