#!/bin/bash

PROGRAM=test_nest_speed_and_memory.py

current=$(date '+%y%m%d%H%M%S')
output_file_dat='memory_consumption_logs/stat'$current'_'$PROGRAM'.dat'
echo "Saving to $output_file_dat"

dstat -tm 10 > $output_file_dat &

ID=$!

echo "Process $ID"
python $PROGRAM '0'
python $PROGRAM '4'
python $PROGRAM '3'
python $PROGRAM '2'
python $PROGRAM '1'

kill -9 $ID
echo "Killed process $ID"




