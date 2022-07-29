#!/bin/sh

data_path="./isicdata/datasets/doctor_case.csv"

# Loading script arguments 
while getopts "nc:ac:fc:r:e:p:" flag; do
    case "${flag}" in
        p) data_path=${OPTARG};;
    esac
done

python server_new.py &
sleep 7 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect

python client_new.py --path ./isicdata/datasets/doctor_case2.csv &
python client_new.py --path $data_path

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`