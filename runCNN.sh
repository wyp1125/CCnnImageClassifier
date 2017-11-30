#!/bin/bash

usage() {
cat << EOF

usage: $0 options

This script runs a customized CNN model to distinguish images with different classes of objects.
A configuration json file should be supplied. 

OPTIONS:
   -h      help, Show this message
   -i      configuration json file
EOF
}

if [ "$#" -eq "0" ]; then
  usage;
  exit;
fi

while getopts "hi:" OPTION
do
     case $OPTION in
         h) usage ; exit 1 ;;
         i) json=$OPTARG ;;
         ?) usage ; exit ;;
     esac
done

if [ ! -f $json ]; then
  echo "The configuration json file does not exist!";
  exit;
fi

PYTHON3=/usr/bin/python3

echo "Activate tensorflow"

tf_activate=/root/tensorflow/bin/activate
source ${tf_activate}

#echo $json
${PYTHON3} train_cnn.py $json
