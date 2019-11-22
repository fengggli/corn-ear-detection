#!/usr/bin/env bash

# First created: 
# Last modified: 2018 Jan 26

# Author: Feng Li
# email: fengggli@yahoo.com
tar  -acv --exclude='__pycache__' \
  -f competition2_stage2.tar.gz \
  data/examples/ \
  data/saved_models/ \
  data/predict.jpg \
  readme.md \
  tests/
  
