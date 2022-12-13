#!/bin/bash

cp $1 z$1.bck 

#Running standard (Hx sweep, mz+)
echo '########################## Running Hx sweep, mz+ #################'
python3 $1 $2 $3
cp $1 $1.Hxmzp.bck

#Running           (Hy sweep, mz+)
echo '########################## Running Hy sweep, mz+ #################'
sed -i 's/np.array(\[i,0,0\])/np.array(\[0,i,0\])/g' $1
sed -i 's/Hx_mzp/Hy_mzp/g' $1
python3 $1 $2 $3
cp $1 $1.Hymzp.bck

#Running           (Hy sweep, mz-)
echo '########################## Running Hy sweep, mz- #################'
sed -i 's/np.array(\[0,0,1\])/np.array(\[0,0,-1\])/g' $1
sed -i 's/Hy_mzp/Hy_mzm/g' $1
python3 $1 $2 $3
cp $1 $1.Hymzm.bck

#Running           (Hx sweep, mz-)
echo '########################## Running Hx sweep, mz- #################'
sed -i 's/np.array(\[0,i,0\])/np.array(\[i,0,0\])/g' $1
sed -i 's/Hy_mzm/Hx_mzm/g' $1
python3 $1 $2 $3
cp $1 $1.Hxmzm.bck

cp z$1.bck $1


