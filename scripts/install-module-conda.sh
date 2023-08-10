#!/bin/sh

# Setting up base directory
BASE_DIR=$(cd .. "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

#####################
# Functions Section #
#####################

ShowHelp() {
    echo "Usage:"
    echo "  $0 [-y] <module-source-dir> <nest-version> <nest-install-dir> <path-conda-setup>"
    echo
    echo "Arguments:"
    echo "  <module-source-dir>  Source path of module"
    echo "  <nest-version>      Version of NEST to use"
    echo "  <nest-install-dir>  Installation directory for NEST"
    echo "  <path-conda-setup>  Path to conda setup script. Ensure conda is available in the"
    echo "                      script path. Usually references the conda.sh in .bashrc or .bash_profile."
    echo
    echo "Options:"
    echo "  -y       Proceed without prompting for confirmation"
    echo "  --help   Display this help and exit"
    exit 1
}

#########################
# Argument Parsing      #
#########################

CONFIRM="true"
while true; do
    case "$1" in
        -y) CONFIRM="false"; shift ;;
        --help) ShowHelp ;;
        *) break ;;
    esac
done

if [[ "$#" -lt 3 ]]; then
    ShowHelp
fi

# Variables based on arguments
MODULE_SOURCE_DIR=$1
NEST_INSTALL_DIR=$3
PATH_CONDA_SETUP=$4

#########################
# Main Script Execution #
#########################

echo -e "\nInitializing conda"
source "$PATH_CONDA_SETUP"

if [ ! -d "$MODULE_SOURCE_DIR" ]; then
    echo "Missing source directory: $MODULE_SOURCE_DIR."
    echo "Do you have the module source code for this version of nest?"
    exit 1
fi

MODULE_DIR=$(basename "$MODULE_SOURCE_DIR")
START=$(date +%s)  # Start time watch
noProcs=$(grep -c 'model name' /proc/cpuinfo)

# Define directories and logs
SOURCE_DIRECTORY="$MODULE_SOURCE_DIR"
BUILD_DIRECTORY="$BASE_DIR/nest/module/build/$MODULE_DIR/"
LOG_DIRECTORY="$BASE_DIR/nest/module/log/"
logFileMake="${LOG_DIRECTORY}${MODULE_DIR}-make"
logFileInstall="${LOG_DIRECTORY}${MODULE_DIR}-install"

# Output paths and instructions
# Output paths and instructions
echo ""
echo "Installing module '$MODULE_NAME'"
echo ""
echo "Clear previous directory and create new one:"
echo "  Build dir:  $BUILD_DIRECTORY"
echo "  Source dir: $SOURCE_DIRECTORY"
echo "  Log dir:    $LOG_DIRECTORY"
echo "  Nest dir:   $NEST_INSTALL_DIR"
echo "  Log files:"
echo "  - $logFileMake"
echo "  - $logFileInstall"
echo ""

[[ "$CONFIRM" == "true" ]] && read -p "Press [Enter] key to continue..."

# Prepare directories
[ -d "$BUILD_DIRECTORY" ] && echo "Removing old build dir $BUILD_DIRECTORY" && rm -r "$BUILD_DIRECTORY"
mkdir -p "$LOG_DIRECTORY" "$BUILD_DIRECTORY"

echo -e "\nStarting installation in $BUILD_DIRECTORY"
cd "$BUILD_DIRECTORY"

# Activate conda environment
conda activate "$NEST_INSTALL_DIR"

# Build and Install
cmake -Dwith-optimize=ON -Dwith-warning=ON -Dwith-nest="$NEST_INSTALL_DIR/bin/nest-config" "$SOURCE_DIRECTORY"
make -j "$noProcs" 2>&1 | tee "$logFileMake"
make -j "$noProcs" install 2>&1 | tee "$logFileInstall"

# Report installation time
END=$(date +%s)
DIFF=$(( END - START ))
echo -e "\nInstallation took $DIFF seconds"

# Test the installation
MODELS="['izhik_cond_exp', 'my_aeif_cond_exp']"
if python -c "import nest; nest.Install('ml_module');models=nest.node_models;print([m for m in models if m in $MODELS]);"; then
    echo -e "\nSuccess!"
else
    echo -e "\nFailed"
fi
echo ""