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
count=[([0]*5)for i in range(1000)]

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/test_for_diff_model/googleNet_0/deploy.prototxt',
                caffe_root + 'models/test_for_diff_model/googleNet_0/bvlc_googlenet_iter_4400000.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB 
# set net to batch size of 50
net.blobs['data'].reshape(1,3,224,224)
for i in xrange(50000):
  # pdb.set_trace()
  line=linecache.getline(TEST,i+1)
  name=line.split( )[0]
  target=int(line.split( )[1])
  count[target][0]+=1
  label2=line.strip('\n')
  input_image = caffe.io.load_image(IMAGE_PATH+name)
  net.blobs['data'].data[...] = transformer.preprocess('data',input_image)
  out = net.forward()
  #print("Predicted class is #{}.".format(out['prob'][0]))

  preDict={}
  for i in xrange(len(out['prob'][0])):
	  preDict[out['prob'][0][i]] = i
  value_max=sorted(preDict.keys())[-1]
  num=preDict[value_max]
  if target==num:
    count[target][1]+=1
    count[target][2]+=1
    for i in xrange(4):
      val = sorted(preDict.keys())[-i-2]
      if target==preDict[val]:
        count[target][2]+=1
        break
for i in xrange(1000):
    if count[i][0]==0:
      count[i][3]=0.0
      count[i][4]=0.0
    else:
      count[i][3]=float(count[i][1])/count[i][0]
      count[i][4]=float(count[i][2])/count[i][0]
    print("%d %d %d %d %.3f %.3f" %(i,count[i][0],count[i][1],count[i][2],count[i][3],count[i][4]))

