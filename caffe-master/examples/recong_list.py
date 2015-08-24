import numpy as np
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import linecache
TEST = '/home/scs4850/Workspace/caffe-master/data/ilsvrc12/val.txt'
import caffe
import os
IMAGE_PATH='/home/scs4850/DataSets/ILSVRC2012_backup/ILSVRC2012_val/'
caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/test_for_diff_model/vgg_spp_0/deploy.prototxt',
                caffe_root + 'models/test_for_diff_model/vgg_spp_0/vgg_spp_finetune_iter_290000.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.float32([104.0, 117.0, 123.0])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB "/n01514668_21.JPEG"
# set net to batch size of 50
net.blobs['data'].reshape(1,3,224,224)
for i in xrange(len(open(TEST,'rU').readlines())):
  line=linecache.getline(TEST,i+1)
  name=line.split( )[0]
  category=int(line.split( )[1])
  image_infor=line.strip('\n')
  input_image = caffe.io.load_image(IMAGE_PATH+name)
  net.blobs['data'].data[...] = transformer.preprocess('data',input_image)
  out = net.forward()
  #print("Predicted class is #{}.".format(out['prob'][0]))
  print category,
  for i in xrange(1000):
    print("%.6f"%(out['prob'][0][i]*100)),
  print '\n',
