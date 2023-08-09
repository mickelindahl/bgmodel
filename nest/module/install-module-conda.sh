#!/bin/sh

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
TEMP=$(getopt -o h: --long help,module-name:,nest-version:,nest-install-dir: -n 'javawrap' -- "$@")
eval set -- "$TEMP"

MODULE_NAME=
NEST_VERSION=
NEST_INSTALL_DIR=
while true; do
  # echo "Current positional argument: $1"
  case "$1" in
  -h | --help)
    Help
    return
    ;;
  --module-name)
    MODULE_NAME="$2"
    shift 2
    ;;
  --nest-version)
    NEST_VERSION="$2"
    shift 2
    ;;
  --nest-install-dir)
    NEST_INSTALL_DIR="$2"
    shift 2
    ;;
  --)
    shift
    break
    ;;
  *) break ;;
  esac
done

if [ -z "$MODULE_NAME" ]; then
    echo "Need to provide --module-name. See ${BASH_SOURCE[0]} --help"
    return
fi

if [ -z "$NEST_VERSION" ]; then
    echo "Need to provide --nest-version. See ${BASH_SOURCE[0]} --help"
    return
fi

if [ -z "$NEST_INSTALL_DIR" ]; then
    echo "Need to provide --nest-install-dir. See ./${BASH_SOURCE[0]} --help"
    return
fi

MODULE_DIR="$MODULE_NAME-$NEST_VERSION"

#Start time watch
START=$(date +%s)

#Get number of processors on the system
noProcs=$(grep -c 'model name' /proc/cpuinfo) 

#Source directory
SOURCE_DIRECTORY="$SCRIPT_DIR/source/$MODULE_DIR/"

#Build directory
BUILD_DIRECTORY="$SCRIPT_DIR/build/$MODULE_DIR/"

#Log directory
LOG_DIRECTORY="$SCRIPT_DIR/log/"
logFileMake="$LOG_DIRECTORY$MODULE_DIR-make"
logFileInstall="$LOG_DIRECTORY$MODULE_DIR-install"
echo ""
echo "Install module '$MODULE_NAME'"
echo ""
echo "Clear previous directory and create new one"
echo ""
echo "Build dir:  $BUILD_DIRECTORY"
echo "Source dir: $SOURCE_DIRECTORY"
echo "Log dir:    $LOG_DIRECTORY"
echo "Nest dir:   $NEST_INSTALL_DIR"
echo ""
echo "With log files:"
echo "$logFileMake"
echo "$logFileInstall"
echo ""
echo "Press [Enter] key to continue..."
read TMP

#Copy source to bootstrap directory

if [ ! -d "$SOURCE_DIRECTORY" ];
then
    echo "Missing source director $SOURCE_DIRECTORY. Do you have module source code for this version of nest?"
    return
fi


if [ -d "$BUILD_DIRECTORY" ];
then
    echo "Removing old build dir $BUILD_DIRECTORY"
    rm -r "$BUILD_DIRECTORY"
fi

echo "Create log dir if it does not exist $LOG_DIRECTORY"
mkdir -p "$LOG_DIRECTORY"

echo "Creating build directory if it does not exist $BUILD_DIRECTORY"
mkdir -p "$BUILD_DIRECTORY"

#echo "Copy source to build dir"
#cp -r "$SOURCE_DIRECTORY" "$BUILD_DIRECTORY"

echo "Start installation."
echo "Press [Enter] key to continue..."
read TMP

#Go into build dir and run cmake
echo "Enter build dir $BUILD_DIRECTORY"
cd "$BUILD_DIRECTORY"

ActivateCondaEnv "$NEST_INSTALL_DIR"

cmake \
    -Dwith-optimize=ON -Dwith-warning=ON \
    -Dwith-nest="$NEST_INSTALL_DIR"/bin/nest-config \
    "$SOURCE_DIRECTORY"

# Make and make install
make -j "$noProcs" 2>&1 | tee "$logFileMake"
make -j "$noProcs" install 2>&1 | tee "$logFileInstall"

#Stop time watch
END=$(date +%s)
DIFF=$(( $END - $START ))

# Move out
cd $SCRIPT_DIR

#Display script execution time
echo "It took $DIFF seconds"
echo ""
echo "Testing import of module $MODULE_NAME"

MODELS="['pif_psc_alpha', 'izhik_cond_exp', 'my_aeif_cond_exp']"

if python -c "import nest; nest.Install('$MODULE_NAME');models=nest.node_models;print([m for m in models if m in $MODELS]);"; then
  echo ""
  echo "Success!"
else
  echo ""
  echo "Failed"
fi
echo ""
