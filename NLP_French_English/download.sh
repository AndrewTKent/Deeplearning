#!/bin/bash

#this downloads the zip file that contains the data
curl http://cs.brown.edu/courses/csci1470/hw_data/hw4.zip --output hw4.zip
# this unzips the zip file - you will get a directory named "data" containing the data
unzip hw4.zip
# this cleans up the zip file, as we will no longer use it
rm hw4.zip

echo downloaded data
