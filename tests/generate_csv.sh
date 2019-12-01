#!/usr/bin/env bash
FILES=/share/Competition2/cornvideos/4inch/*.MP4
# cannot read 2inch video in sievert, why?

for file in $FILES
do
  echo "process video $file"
  #python3 tests/detect.py  --videopath=$file --stepsize 10
  python3 tests/detect.py  --videopath=$file --stepsize 1 --norender
done
