#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BASE_DIR=$(dirname "$SCRIPT_DIR")

#####################
# Functions Section #
#####################

# Function to create a conda environment
CreateCondaEnv() {
  local CONDA_ENV_PATH=$1
  local INSTALL_DIRECTORY_CONDA_CACHE=$2
  local ENVIRONMENT_FILE="$BASE_DIR/environment.yml"

  echo -e "\n\033[1mSetting up Conda environment...\033[0m"

  if [ -d "$INSTALL_DIRECTORY_CONDA_CACHE" ]; then
    echo "Conda CACHE env found at $INSTALL_DIRECTORY_CONDA_CACHE."
    echo "Will use it as conda environment at $CONDA_ENV_PATH"
    cp -r "$INSTALL_DIRECTORY_CONDA_CACHE" "$CONDA_ENV_PATH"
  else
    echo "Creating new conda environment..."
    conda env create --debug -p "$CONDA_ENV_PATH" --file "$ENVIRONMENT_FILE"
    echo "Copying environment to cache..."
    cp -r "$CONDA_ENV_PATH" "$INSTALL_DIRECTORY_CONDA_CACHE"
  fi

  echo -e "\033[32mConda setup complete.\033[0m\n"
}

# Function to download NEST source files
DownloadNestToSourceDirectory() {
  local NEST_FOLDER_NAME=$1
  local NEST_TAR=$2
  local SOURCE_DIRECTORY=$3
gnore
  echo -e "\033[1mDownloading NEST source files...\033[0m"

  if [ -d "$SOURCE_DIRECTORY" ]; then
    echo "Source files already exist at $SOURCE_DIRECTORY"
  else
    mkdir -p $(dirname SOURCE_DIRECTORY)
    echo "Moving to source directory..."
    cd source
    local URL="https://github.com/nest/nest-simulator/archive/refs/tags/$NEST_TAR"
    echo "Downloading NEST from $URL"
    wget "$URL"
    echo "Extracting $NEST_TAR to source directory..."
    tar -zxvf "$NEST_TAR"
    cd ..
  fi

  echo -e "\033[32mNEST download complete.\033[0m\n"
}

# Function to display script usage
ShowHelp() {
    echo -e "\033[1mUsage:\033[0m"
    echo "  $0 [-y] [--ignore-conda-cache] <nest-version> <path-conda-setup>\n"
    echo -e "\033[1mArguments:\033[0m"
    echo "  <nest-version>         Version of NEST to use"
    echo "  <path-conda-setup>     Path to conda setup script. (e.g. conda.sh in .bashrc)"
    echo -e "\n\033[1mOptions:\033[0m"
    echo "  -y                    Skip confirmation prompts"
    echo "  --ignore-conda-cache  Do not use previous install conda environment if exists during installations"
    echo "  --help                Display this help and exit"
    exit 1
}


#########################
# Argument Parsing      #
#########################

# Parsing command line arguments
CONFIRM="true"
IGNORE_CONDA_CACHE="false"
while true; do
    case "$1" in
        -y) CONFIRM="false"; shift ;;
        --ignore-conda-cache) IGNORE_CONDA_CACHE="true"; shift ;;
        --help) ShowHelp ;;
        *) break ;;
    esac
done

# Ensure correct arguments are passed
if [[ "$#" -lt 2 ]]; then
    ShowHelp
fi

# Setting up initial configurations
NEST_VERSION=$1
PATH_CONDA_SETUP=$2


#########################
# Main Script Execution #
#########################

echo -e "\n\033[1mInitializing conda...\033[0m"
source "$PATH_CONDA_SETUP"

NEST_FOLDER_NAME="nest-simulator-$NEST_VERSION"
NEST_TAR="v$NEST_VERSION.tar.gz"
NUMBER_OF_PROCESSORS=$(grep -c 'model name' /proc/cpuinfo)

# Defining various directories for setup
SOURCE_DIRECTORY="$BASE_DIR/nest/dist/source/$NEST_FOLDER_NAME/"
BUILD_DIRECTORY="$BASE_DIR/nest/dist/build/$NEST_FOLDER_NAME/"
INSTALL_DIRECTORY="$BASE_DIR/nest/dist/install/$NEST_FOLDER_NAME"
INSTALL_DIRECTORY_CONDA_CACHE="$INSTALL_DIRECTORY-conda-cache"
LOG_DIRECTORY="$BASE_DIR/nest/dist/log/"

# Proceeding with NEST setup
echo -e "\n\033[1mSetting up NEST...\033[0m"
DownloadNestToSourceDirectory "$NEST_FOLDER_NAME" "$NEST_TAR" "$SOURCE_DIRECTORY"

# Prepare for build
echo -e "\n\033[1mPreparing for build...\033[0m"
echo "Clearing previous installations and build directories..."
if [[ "$CONFIRM" == "true" ]]; then
  read -p "Press [Enter] key to continue..." TMP
fi

if [ -d "$BUILD_DIRECTORY" ]; then
  echo -e "\nRemoving $BUILD_DIRECTORY"
  rm -r "$BUILD_DIRECTORY"
else
  echo -e "\nNo previous build dir to remove"
fi

echo "Creating build dir $BUILD_DIRECTORY"
mkdir -p "$BUILD_DIRECTORY"

echo "Removing $INSTALL_DIRECTORY"
if [ -d "$INSTALL_DIRECTORY_CONDA_CACHE" ]; then
   echo "Cache found"
   if [ "$IGNORE_CONDA_CACHE" = "true" ]; then
     echo "Remove both install and cache directory"
     conda deactivate
      conda env remove -y -p "$INSTALL_DIRECTORY"
      rm -r "$INSTALL_DIRECTORY_CONDA_CACHE"
   elif [ -d "$INSTALL_DIRECTORY"  ];then
     echo "Remove install directory"
      rm -r "$INSTALL_DIRECTORY"
   fi
else
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
if [[ "$CONFIRM" == "true" ]]; then
  read -p "Press [Enter] key to continue..." TMP
fi

#Start time watch
START=$(date +%s)

# Python with conda
CreateCondaEnv "$INSTALL_DIRECTORY" "$INSTALL_DIRECTORY_CONDA_CACHE"

echo ""
conda activate "$INSTALL_DIRECTORY"

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

echo "conda activate $INSTALL_DIRECTORY" > "$BASE_DIR/conda-activate.sh"
chmod +x "$BASE_DIR/conda-activate.sh"
cd "$BASE_DIR"

echo ""
#Display script execution time
echo -e "\033[32mDone!\033[0m"
echo "---------------------------------------------"
echo ""
echo -e "\033[1mDirectories:\033[0m"
echo -e "  \033[36mBuild:\033[0m    $BUILD_DIRECTORY"
echo -e "  \033[36mInstall:\033[0m  $INSTALL_DIRECTORY"
echo -e "  \033[36mLog:\033[0m      $LOG_DIRECTORY"
echo ""
echo "---------------------------------------------"
echo -e "Installation took \033[33m$DIFF_TOTAL\033[0m seconds"
echo "---------------------------------------------"
echo ""
echo -e "\033[1mActivation:\033[0m"
echo -e "1. Using \033[34mconda\033[0m command:"
echo -e "   \033[36mconda activate $CONDA_ENV_PATH\033[0m"
echo ""
echo -e "2. Using provided script:"
echo -e "   \033[36msource ./conda-activate\033[0m"
echo ""
echo "---------------------------------------------"
echo -e "\033[1mEditor Setup:\033[0m"
echo -e "Set Python environment in your editor to \033[36m$CONDA_ENV_PATH\033[0m"
echo ""
echo "---------------------------------------------"
echo -e "Conda environment was created from file:"
echo -e "   \033[34m$ENVIRONMENT_FILE\033[0m"
echo ""
echo "---------------------------------------------"
echo "You should now be able to import nest after activating conda environment"