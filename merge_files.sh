#!/usr/bin/bash


for file in $(find data/unpacked/ -name '*.sgf')
do
    #echo $file
    echo $(cat $file | tr -d '\n') >> 'data/full_file.txt'
done

sed -i '/^$/d' 'full_file.txt'
