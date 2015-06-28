#!/bin/bash

current=$(date '+%y%m%d%H%M%S')
#output_file_csv='memory_consumption_logs/stat'$current'.csv'
output_file_dat='memory_consumption_logs/stat'$current'_'$1'_'$2'.dat'
echo "Saving to $output_file_dat"
#rm $output_file

# With nohup then 
#ID=nohup dstat -tcmn 10 > dstat.dat &
#dstat -tm --output $output_file_csv 10 > $output_file_dat &
dstat -tm 10 > $output_file_dat &

ID=$!

#watch 'grep VmSize /proc/PID/status >> log'
echo "Process $ID"
#./job.sh
python $1
kill -9 $ID
echo "Killed process $ID"
#The following script will search for lines containing the 
# terms ‘time’ and ‘date’, delete these lines and save the 
# output in a new file called ‘stat.dat’.

#grep -Ev "time|system" dstat.dat > stat.dat




