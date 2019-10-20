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
}

echo  Append:
echo "     `pwd`/TensorFlow/models/research/:`pwd`/TensorFlow/models/research/slim/" 
echo TO PYTHONPATH!

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
}

# compile model
{
  PROTOC_BIN=`pwd`/protobuf/bin/protoc
  cd TensorFlow/models/research/
  $PROTOC_BIN object_detection/protos/*.proto --python_out=.
  echo "Compiled! using protoc"
}
