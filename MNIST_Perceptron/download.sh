#!/bin/bash

#this downloads the zip file that contains the data
curl http://cs.brown.edu/courses/csci1470/hw_data/hw1.zip --output hw1.zip
# this unzips the zip file - you will get a directory named "data" containing the data
unzip hw1.zip
# this cleans up the zip file, as we will no longer use it
rm hw1.zip

echo downloaded data
