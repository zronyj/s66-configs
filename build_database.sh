#!/bin/bash

tar -xvf DS_0j2smy6relq0_0.tar.xz
#rm DS_0j2smy6relq0_0.tar.xz

mkdir s66

cd nenci2021/xyzfiles
for f in {001..066}
do
cp $(ls *.xyz | grep -v '_[AB]_' | grep $f) ../../s66
done

cd ../../

mkdir md
cp md_runner.py md
mkdir aligned
cp traj_reader.py aligned
mkdir filtered
cp struct_filter.py filtered