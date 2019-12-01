#!/usr/bin/env bash
FILES=/share/Competition2/cornvideos/4inch/*.MP4

for file in $FILES
do
  echo "process video $file"
  python3 tests/detect.py  --videopath=$file --stepsize 10
done
