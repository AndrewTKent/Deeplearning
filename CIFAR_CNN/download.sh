#!/bin/bash

#this downloads the zip file that contains the data
curl http://cs.brown.edu/courses/csci1470/hw_data/hw2.zip --output hw2.zip
# this unzips the zip file - you will get a directory named "data" containing the data
unzip hw2.zip
# this cleans up the zip file, as we will no longer use it
rm hw2.zip

echo downloaded data
