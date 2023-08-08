#!/bin/bash

##########################################################
# ActivateCondaEnv                                       #
##########################################################
ActivateCondaEnv() {

  CONDA_ENV=$1

  echo "Activate conda environment $CONDA_ENV"
  #conda init bash
  conda activate "$CONDA_ENV"
}
##########################################################
# CheckIfRunAsSource                                     #
##########################################################
CheckIfRunAsSource() {
  if [[ ! "${BASH_SOURCE[0]}" != "${0}" ]]; then

    echo "script needs to be run with source as: 'source ${BASH_SOURCE[0]} ...'"
    exit

  fi
}
############################################################
############################################################
# CreateCondaEnv                                           #
############################################################
CreateCondaEnv() {

  CONDA_ENV=$1
  PYTHON_VERSION=$2

  echo "Create conda environment $CONDA_ENV with pyton $PYTHON_VERSION using nest environment.yml ."
  echo "Please select y if you have not done this previous (y/N)"
  read -r CHOICE
  case $CHOICE in
  y | Y)
    conda remove -y --name $CONDA_ENV --all
    conda create -y --name $CONDA_ENV --file environment.yml python=$PYTHON_VERSION
    ;;
  *) echo "Skipping create conda ..." ;;
  esac
}
###########################################################
###########################################################
# DownloadNestToSourceDirectory                           #
###########################################################
DownloadNestToSourceDirectory() {
  NEST_FOLDER_NAME=$1
  NEST_TAR=$2

  if [ -d "source/$NEST_FOLDER_NAME" ]; then

    echo "Source files already downloaded"

  else

    mkdir -p source

    echo "Entering source folder"
    cd source
    URL="https://github.com/nest/nest-simulator/archive/refs/tags/$NEST_TAR"
    echo "Downloading nest from $URL"
    wget "$URL"

    echo "Unpacking $NEST_TAR to source folder"
    tar -zxvf "$NEST_TAR"
    cd ..
  fi
}
############################################################
############################################################
# Help                                                     #
############################################################
Help() {
  echo ""
  echo "Script for installing nest width conda environment"
  echo ""
  echo "Usage:"
  echo "source install.sh [options]"
  echo "source install.sh -h|-help"
  echo ""
  echo "Options":
  echo " -h, --help            Display help text"
  echo " --nest-version        NEST version to use. Defaults to 3.3"
  echo " --python-version      Python version to use. Defaults to 3.9"
  echo ""
}
###########################################################
###########################################################
# InstallOsDependencies                                   #
###########################################################
InstallOsDependencies() {
  echo "Install OS dependencies (y/N)?"
  read -r CHOICE
  case $CHOICE in
  y | Y) sudo apt install -y \
    libgsl-dev \
    libltdl-dev \
    libncurses-dev \
    libreadline-dev \
    openmpi-bin \
    libopenmpi-dev ;;
  *) echo "Skipping OS dependencies..." ;;
  esac
}
###########################################################
###########################################################
# InstallExtraPythonDependencies                          #
###########################################################
InstallExtraPythonDependencies() {

  CONDA_ENV=$1

  echo "Install extra python dependencies (y/N)"
  read -r CHOICE
  case $CHOICE in
  y | Y)
    conda install -y -n "$CONDA_ENV" pip
    conda install -c anaconda psycopg2
    conda run -n "$CONDA_ENV" pip install NeuroTools
    echo "Done!"
    ;;
  *) echo "Skipping python dependencies ..." ;;
  esac
}
########################################################
########################################################
# Main program                                         #
########################################################

CheckIfRunAsSource

ARGS="$@"
TEMP=$(getopt -o h: --long help,nest-version:,python-version: -n 'javawrap' -- "$@")
eval set -- "$TEMP"

NEST_VERSION=3.3
PYTHON_VERSION=3.9
while true; do
  # echo "Current positional argument: $1"
  case "$1" in
  -h | --help)
    Help
    exit
    ;;
  --nest-version)
    NEST_VERSION="$2"
    shift 2
    ;;
  --python-version)
    PYTHON_VERSION="$2"
    shift 2
    ;;
  --)
    shift
    break
    ;;
  *) break ;;
  esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

NEST_FOLDER_NAME=nest-simulator-$NEST_VERSION
NEST_TAR=v$NEST_VERSION.tar.gz
CONDA_ENV="nest-$NEST_VERSION"

# OS
InstallOsDependencies

# Python
CreateCondaEnv "$CONDA_ENV" "$PYTHON_VERSION"
InstallExtraPythonDependencies "$CONDA_ENV"
ActivateCondaEnv "$CONDA_ENV"

# NEST
DownloadNestToSourceDirectory "$NEST_FOLDER_NAME" "$NEST_TAR"
#CMAKE_TAG="3.23.2"
#CMAKE_TAG="3.16.9"
#CMAKE_NAME="cmake-$CMAKE_TAG-Linux-x86_64"
#CMAKE_TAR_GZ="cmake-$CMAKE_TAG-linux-x86_64.tar.gz"
#CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v$CMAKE_TAG/$CMAKE_TAR_GZ"
#echo "$SCRIPT_DIR"

#echo "Download cmake $CMAKE_TAG in (y/N)?"
#read -r CHOICE
#case $CHOICE in
#   y | Y) if [ -d ./$CMAKE_NAME ];
#             then echo "cmake binary already downloaded ...";
#          else
#             wget $CMAKE_URL;
#             tar -zxvf "$CMAKE_TAR_GZ";
#          fi;;
#    *) echo "Skipping cmake...";;
#esac

echo "Proceeding with installation"
echo
#Start time watch
START=$(date +%s)

NUMBER_OF_PROCESSORS=$(grep -c 'model name' /proc/cpuinfo)

CURRENT_DIRECTORY=$(pwd)

BUILD_DIRECTORY="$CURRENT_DIRECTORY/build/$NEST_FOLDER_NAME/"
LOG_DIRECTORY="$CURRENT_DIRECTORY/log/"
INSTALL_DIRECTORY="$CURRENT_DIRECTORY/install/$NEST_FOLDER_NAME/"
SOURCE_DIRECTORY="$CURRENT_DIRECTORY/source/$NEST_FOLDER_NAME/"

#Remove old build
echo ""
echo "Clear previous installation and build directories"
echo "Build dir: $BUILD_DIRECTORY"
echo "Install dir: $INSTALL_DIRECTORY"
echo "Log dir: $LOG_DIRECTORY"
echo "Press [Enter] key to continue..."
read TMP

if [ -d "$BUILD_DIRECTORY" ]; then
  echo "Removing $BUILD_DIRECTORY"
  rm -r "$BUILD_DIRECTORY"

else
  echo "No previous build dir to remove"
fi

echo "Creating build dir $BUILD_DIRECTORY"
mkdir -p "$BUILD_DIRECTORY"

if [ -d "$INSTALL_DIRECTORY" ]; then
  echo "Removing $INSTALL_DIRECTORY"
  rm -r "$INSTALL_DIRECTORY"
else
  echo "Not previous install dir to remove"
fi

echo "Creating install dir $INSTALL_DIRECTORY"
mkdir -p "$INSTALL_DIRECTORY"

echo "Create log dir if it does not exist $LOG_DIRECTORY"
mkdir -p "$LOG_DIRECTORY"

echo "Enter build dir $BUILD_DIRECTORY"
cd "$BUILD_DIRECTORY"

echo ""
echo "Press [Enter] key to continue..."
read TMP

echo ""
echo "Cmake"
#conda run --live-stream -n $CONDA_ENV $SCRIPT_DIR/$CMAKE_NAME/bin/cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIRECTORY" "$SOURCE_DIRECTORY"
cmake -Dwith-python=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_DIRECTORY" "$SOURCE_DIRECTORY" 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-cmake"
echo ""
echo "Make"
#conda run --live-stream -n $CONDA_ENV make -j "$NUMBER_OF_PROCESSORS" 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-make"
make 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-make"
echo ""
echo "Make install"
#conda run --live-stream -n $CONDA_ENV make install -d 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-install"
make install 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-make-install"
echo ""
echo "Make installcheck"
#conda run --live-stream -n $CONDA_ENV make installcheck
make installcheck 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-make-installcheck"
#Stop time watch
END=$(date +%s)
DIFF=$(($END - $START))

cd "$SCRIPT_DIR"

#Display script execution time
echo "It took $DIFF seconds"
