#!/bin/bash

current=$(date '+%y%m%d%H%M%S')
output_file_dat='memory_consumption_logs/stat'$current'_'$1'_'$2'.dat'
echo "Saving to $output_file_dat"

dstat -tm 10 > $output_file_dat &

ID=$!

echo "Process $ID"
# Seems i can not pipe here becaus then memory logs gets blank
python $1 

kill -9 $ID
echo "Killed process $ID"




