#!/usr/bin/env bash
# instruction: run with ./process_data.sh BOX_DIR
if [ $# -eq 1 ] && [ -d $1/annotation_xml ]; then
  BOX_DIR=$1
  echo box dir is $BOX_DIR
else
  echo "You need to run: export BOX_DIR=path-to-Competition2"
  exit
fi
#/share/Competition2/fromBox

NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)
SOURCE_DIR=$(pwd)

ANNOTATION_DIR=$BOX_DIR/annotation_xml
IMAGES_DIR=$BOX_DIR/images_original

OUTPUT_DIR=/tmp/cornear-$NEW_UUID

mkdir -pv $OUTPUT_DIR
cd $OUTPUT_DIR

# collect all xml 
{
  mkdir -pv annotations/train
  mkdir -pv annotations/test

  mkdir -pv all_xml 
  cp $ANNOTATION_DIR/*revised/*.xml all_xml #325 xmls, some don't have connections

  grep -Rl "<name>1</name>" all_xml/ | shuf &> selected.list #215 contains "1"
  num_val=15 #15 samples for validation
  cat selected.list |head -n -$num_val | xargs -I % cp % annotations/train
  cat selected.list |tail -n $num_val | xargs -I % cp % annotations/test
}


# collect all images
{
  mkdir -pv images
  cp $IMAGES_DIR/*/*.jpg  images #346 imgs
}

# generate xml
python3 $SOURCE_DIR/xml_to_csv.py annotations/

echo "now generating tfrecords..."
# generate tf_records
python3 $SOURCE_DIR/generate_tfrecord.py --csv_input=annotations/train_labels.csv --image_dir=images --output_path=images/train-1class.record
python3 $SOURCE_DIR/generate_tfrecord.py --csv_input=annotations/test_labels.csv --image_dir=images --output_path=images/test-1class.record


echo "All saved in $OUTPUT_DIR/"
