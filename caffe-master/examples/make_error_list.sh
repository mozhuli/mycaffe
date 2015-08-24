#!/usr/bin/env sh
# make top1 error list,image from IMAGE_PATH,
# N.B. this is available in data/ilsvrc12

IMAGE_PATH=/home/scs4850/DataSets/ILSVRC2012_backup/ILSVRC2012_val/
TXT_PATH=/home/scs4850/Workspace/caffe-master/data/ilsvrc12/val.txt
DEPLOY_PATH=models/test_for_diff_model/googleNet_0/deploy.prototxt
MODLE_PATH=models/test_for_diff_model/googleNet_0/bvlc_googlenet_iter_4400000.caffemodel
OUT_PATH=
EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/ilsvrc12_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
