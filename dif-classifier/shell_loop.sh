#!/bin/bash

FILES=/path/to/*.tif
for f in $FILES
do
  echo "Processing $f file..."
  # Perform tracking on each file and output xml FILES
  #./ImageJ-linux64 --ij2 --headless --run ~/source/diff-classifier/dif-classifier/example_trackmate_script.py
  # This command needs to be modified to make the file an input so I can loop over it.
