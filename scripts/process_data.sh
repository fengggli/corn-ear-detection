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
TFRECORD_DIR=/share/Competition2/images

NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)
SOURCE_DIR=$(pwd)

# 4 inch data
ANNOTATION_DIR=$BOX_DIR/annotation_xml
IMAGES_DIR=$BOX_DIR/images_original

#2 inch data and annotations
INCH2_DIR=$BOX_DIR/2inch

OUTPUT_DIR=/tmp/cornear-$NEW_UUID

mkdir -pv $OUTPUT_DIR
cd $OUTPUT_DIR

# collect all xml 
{
  mkdir -pv annotations/train
  mkdir -pv annotations/test

  mkdir -pv all_xml 
  cp $ANNOTATION_DIR/*xml/*.xml all_xml #325 xmls, some don't have connections
  cp $INCH2_DIR/*xml/*.xml all_xml #325 xmls, some don't have connections

  grep -Rl "<name>[12]</name>" all_xml/ | shuf &> selected.list #xml 439(290 for 4inch)contains "1" or "2"
  num_val=45 #10% samples for validation
  cat selected.list |head -n -$num_val | xargs -I % cp % annotations/train
  cat selected.list |tail -n $num_val | xargs -I % cp % annotations/test
}


# collect all images
{
  mkdir -pv images
  cp $IMAGES_DIR/*/*.jpg  images #346 imgs
  cp $INCH2_DIR/*/*.jpg  images #346 imgs
}

# generate xml
python3 $SOURCE_DIR/xml_to_csv.py annotations/

echo "now generating tfrecords..."
# generate tf_records
python3 $SOURCE_DIR/generate_tfrecord.py --csv_input=annotations/train_labels.csv --image_dir=images --output_path=images/train-2class.record
python3 $SOURCE_DIR/generate_tfrecord.py --csv_input=annotations/test_labels.csv --image_dir=images --output_path=images/test-2class.record


echo "All saved in $OUTPUT_DIR/"
if [ -d /share/Competition2 ]; then # only in sievert
  rm -rf $TFRECORD_DIR
  cp -r $(readlink -f images) $TFRECORD_DIR
  cp -r $(readlink -f annotations) $TFRECORD_DIR/images
  echo "saving annotations from $(readlink -f images) to $TFRECORD_DIR" 
fi

