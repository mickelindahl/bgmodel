#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

##########################################################
# ActivateCondaEnv                                       #
##########################################################
ActivateCondaEnv() {

  CONDA_ENV_PATH=$1

  echo "Activate conda environment at $CONDA_ENV_PATH"
  #conda init bash
  conda activate "$CONDA_ENV_PATH"
}
##########################################################
# CheckIfRunAsSource                                     #
##########################################################
CheckIfRunAsSource() {
  if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then

    echo "script needs to be run with source as: 'source ${BASH_SOURCE[0]} ...'"
    echo "because conda activate needs to be run as source"
    exit

  fi
}
############################################################
############################################################
# CreateCondaEnv                                           #
############################################################
CreateCondaEnv() {

  CONDA_ENV_PATH=$1
  CONDA_CACHE_PATH=$2
  ENVIRONMENT_FILE="$SCRIPT_DIR/environment.yml"

  # Python
  if [ -d "$CONDA_CACHE_PATH" ]; then
    echo "Conda CACHE env found at $CONDA_CACHE_PATH."
    echo "Will use it as conda environment at $CONDA_ENV_PATH"
    cp -r "$CONDA_CACHE_PATH" "$CONDA_ENV_PATH"
#    read TMP
  else
    echo "Crate conda environment"
    conda env create --debug -p "$CONDA_ENV_PATH" --file "$ENVIRONMENT_FILE"
    echo "Copy environment to cache"
    cp -r "$CONDA_ENV_PATH" "$CONDA_CACHE_PATH"
  fi
}
###########################################################
###########################################################
# DownloadNestToSourceDirectory                           #
###########################################################
DownloadNestToSourceDirectory() {
  NEST_FOLDER_NAME=$1
  NEST_TAR=$2
  SOURCE_DIRECTORY=$3
  if [ -d "source/$NEST_FOLDER_NAME" ]; then

    echo "Source files already downloaded at $SCRIPT_DIR/source/$NEST_FOLDER_NAME"

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
  echo "Script for installing nest"
  echo ""
  echo "Usage:"
  echo "source ${BASH_SOURCE[0]} [options]"
  echo "source ${BASH_SOURCE[0]} -h|-help"
  echo ""
  echo "Options:"
  echo " -h, --help            Display help text"
  echo " --nest-version        NEST version to use"
  echo " --python-version      Python version to use"
  echo ""
}
###########################################################
########################################################
# Main program                                         #
########################################################

CheckIfRunAsSource

ARGS="$@"
TEMP=$(getopt -o h: --long help,nest-version:,python-version: -n 'javawrap' -- "$@")
eval set -- "$TEMP"

NEST_VERSION=
PYTHON_VERSION=
while true; do
  # echo "Current positional argument: $1"
  case "$1" in
  -h | --help)
    Help
    return
    ;;
  --nest-version)
    NEST_VERSION="$2"
    shift 2
    ;;
  --)
    shift
    break
    ;;
  *) break ;;
  esac
done

if [ -z "$NEST_VERSION" ]; then
    echo "Need to provide --nest-version. See ./${BASH_SOURCE[0]} --help"
    return
fi

NEST_FOLDER_NAME=nest-simulator-$NEST_VERSION
NEST_TAR=v$NEST_VERSION.tar.gz

NUMBER_OF_PROCESSORS=$(grep -c 'model name' /proc/cpuinfo)

BUILD_DIRECTORY="$SCRIPT_DIR/build/$NEST_FOLDER_NAME/"
LOG_DIRECTORY="$SCRIPT_DIR/log/"
INSTALL_DIRECTORY="$SCRIPT_DIR/install/$NEST_FOLDER_NAME"
CONDA_CACHE_PATH="$INSTALL_DIRECTORY-conda-cache"
SOURCE_DIRECTORY="$SCRIPT_DIR/source/$NEST_FOLDER_NAME/"

# NEST
echo ""
DownloadNestToSourceDirectory "$NEST_FOLDER_NAME" "$NEST_TAR" "$SOURCE_DIRECTORY"

#Remove old build
echo ""
echo "Clear previous installation and build directories"
echo ""
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

echo "Removing $INSTALL_DIRECTORY"
if [ -d "$CONDA_CACHE_PATH" ]; then
   # If cache found just remove it. Cache will later be used as environment
   echo "Cache found"
   if [ -d "$INSTALL_DIRECTORY"  ];then
      rm -r "$INSTALL_DIRECTORY"
   fi
else
   # If not cache found make sure to remove conda environment using conda
   echo "No cache found"
   conda deactivate
   conda env remove -y -p "$INSTALL_DIRECTORY"
   rm -r "$CONDA_ENV_PATH" # Ensure it is completely removed so copy will be correct
fi

echo "Create log dir if it does not exist $LOG_DIRECTORY"
mkdir -p "$LOG_DIRECTORY"

echo "Enter build dir $BUILD_DIRECTORY"
cd "$BUILD_DIRECTORY"

echo ""
echo "The conda python environment and nest will be install in:"
echo "    $INSTALL_DIRECTORY."
echo ""
echo "Press [Enter] key to continue..."
read TMP

#Start time watch
START=$(date +%s)

# Python with conda
CreateCondaEnv "$INSTALL_DIRECTORY" "$CONDA_CACHE_PATH"
END_CONDA=$(date +%s)
DIFF_CONDA=$(("$END_CONDA" - "$START"))
echo ""

ActivateCondaEnv "$CONDA_ENV_PATH"

echo ""
echo "Cmake $SOURCE_DIRECTORY"
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIRECTORY \
      $SOURCE_DIRECTORY 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-cmake"
echo ""
echo "Make"
make -j "$NUMBER_OF_PROCESSORS" 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-make"
echo ""
echo "Make install"
make install 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-make-install"
echo ""
echo "Make installcheck"
make installcheck 2>&1 | tee "$LOG_DIRECTORY$NEST_FOLDER_NAME-make-installcheck"

#Stop time watch
END=$(date +%s)
DIFF_TOTAL=$(("$END" - "$START"))
DIFF_NEST=$(("$DIFF_TOTAL" - "$DIFF_CONDA"))

echo "conda activate $CONDA_ENV_PATH" > "$SCRIPT_DIR/conda-activate.sh"
chmod +x "$SCRIPT_DIR/conda-activate.sh"
cd "$SCRIPT_DIR"
echo "CONDA_INSTALLED: $CONDA_INSTALLED"
echo ""
#Display script execution time
echo "Done!"
echo ""
echo "Build dir:   $BUILD_DIRECTORY"
echo "Install dir: $INSTALL_DIRECTORY"
echo "Log dir:     $LOG_DIRECTORY"
echo ""
echo "Conda installation took $DIFF_CONDA seconds"
echo "Nest installation $DIFF_NEST seconds"
echo ""
echo "Activate conda environment with command"
echo "    conda activate $CONDA_ENV_PATH"
echo ""
echo " or"
echo ""
echo "Runscript conda-activate.sh with source in root (source ./conda-activate)"
echo ""
echo " For your editor set python conda environment to $CONDA_ENV_PATH"
echo ""
echo "Conda was created using environment file:"
echo "   $ENVIRONMENT_FILE."
echo ""
echo "You should now be able to import nest"
