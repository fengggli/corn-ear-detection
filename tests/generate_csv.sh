#!/usr/bin/env bash
#FILES=/share/Competition2/cornvideos/4inch/*.MP4
FILES=/share/Competition2/cornvideos/2inch_muted/*.MP4
# cannot read 2inch video in sievert, why? -> see tests/test_cv.py

for file in $FILES
do
  echo "process video $file"
  #python3 tests/detect.py  --videopath=$file --stepsize 10
  python3 tests/detect.py  --videopath=$file --stepsize 1 --norender
done
