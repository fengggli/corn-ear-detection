myroot=`pwd`
{
  if [ -d "TensorFlow" ]; then
    echo "already downloaded tf models, pass"
  else
    wget https://github.com/tensorflow/models/archive/v1.13.0.zip
    mkdir TensorFlow
    cd TensorFlow
    unzip ../v1.13.0.zip
    mv models-* models
    echo "Tf models Downloaded"
  fi

  cd $myroot
}


{
  pkgname="protobuf"
  if [ -d $pkgname ]; then
    echo "$pkgname already installed, pass"
  else
    wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
    mkdir $pkgname
    cd $pkgname
    unzip ../protobuf.zip
  fi
  cd $myroot
}

# compile model
{
  PROTOC_BIN=`pwd`/protobuf/bin/protoc
  cd TensorFlow/models/research/
  $PROTOC_BIN object_detection/protos/*.proto --python_out=.
  echo "Compiled! using protoc"
  $PROTOC_BIN --version
  cd $myroot
}

# coco
{
  echo "cur dir is $PWD"
  TARGETDIR=`pwd`/TensorFlow/models/research/
  if [ -d $TARGETDIR/pycocotools ]; then
    echo "coco installed, pass"
  else
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    make
    cp -r pycocotools $TARGETDIR
    echo "coco installed in $TARGETDIR"
  fi
}

echo  Now run::
echo " source ./setenv.sh"

