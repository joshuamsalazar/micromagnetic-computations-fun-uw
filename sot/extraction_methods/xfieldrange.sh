#!/bin/bash

#Backup original file
echo 1 $1 2 $2  3 $3 4 $4 5 $5
cp $1 z$1.bck
for field in $(seq $4 $5 $6)  
do
    echo '########################## Running Hext ' $field ' #################'
    sed -i 's/HEXT/0.0'$field'/g' $1
    python3 $1 $2 ${field}mT
    sed -i 's/0.0'$field'/HEXT/g' $1
done
#

#Running         



#Replace original file back
cp z$1.bck $1
