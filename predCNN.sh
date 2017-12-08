#!/bin/bash

usage() {
cat << EOF

usage: $0 options

This script uses an established CNN model for different classes of objects to distinguish any new images.
The configuration json file for the CNN model should be supplied. 

OPTIONS:
   -h      help, Show this message
   -i      configuration json file (required)
   -f      folder containing new images for classification (required)
   -o      output file (required)
EOF
}

if [[ "$#" -lt 6 ]]; then
  usage;
  exit;
fi

while getopts "hi:f:o:" OPTION
do
     case $OPTION in
         h) usage ; exit 1 ;;
         i) json=$OPTARG ;;
         f) folder=$OPTARG ;;
         o) out_file=$OPTARG ;;
         ?) usage ; exit ;;
     esac
done

if [ ! -f $json ]; then
  echo "The configuration json file does not exist!";
  exit;
fi

if [ ! -d $folder ]; then
  echo "The folder containing new images does not exist!";
  exit;
fi

PYTHON3=/usr/bin/python3

if ! [ -x "$PYTHON3" ]; then
  echo "Python3 could not be found!";
  exit;
fi

tf_activate=/home/yupeng/tensorflow/bin/activate

if ! [ -f "$tf_activate" ]; then
  echo "TensorFlow could not be found!";
  exit;
fi

source ${tf_activate}

echo "Tensorflow is activated!"

${PYTHON3} predict_cnn.py $json $folder $out_file
