#!/usr/bin/env bash
CALLING_DIR=$PWD
SCRIPT_DIR=$( dirname -- "${BASH_SOURCE[0]}" )
cd $SCRIPT_DIR;
if [ $(command -v python) ]; then
    python=python;
else
    python=python3;
fi
for i in $(cat config.py | awk -F"[= ]"  ' {print $1}'); do 
    if [[ $i != "import" &&  ${i:0:1} != "#" ]] ; then
        tmp=$($python -c "import config; print (config.$i)"); 
        export $i=$tmp ; 
        echo "INFO: $i = $tmp"
    fi;
done
cd $CALLING_DIR