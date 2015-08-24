import numpy as np
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import linecache
import caffe
PATH = '/home/scs4850/Workspace/caffe-master/models/a.txt'
import os
import pdb

count=[([0]*5)for i in range(1000)]
for i in xrange(50000):
  # pdb.set_trace()
  line=linecache.getline(PATH,i+1)
  category=int(float(line.split( )[0]))
  count[category][0]+=1
  preDict={}
  for j in xrange(1000):
	  preDict[float(line.split( )[j+1])] = j
  value_max=sorted(preDict.keys())[-1]
  predict_category=preDict[value_max]
  #pdb.set_trace()
  if category==predict_category:
    count[category][1]+=1
    count[category][2]+=1
  else:
    for i in xrange(4):
      val = sorted(preDict.keys())[-i-2]
      temp_category=preDict[val]
      if category==temp_category:
        count[category][2]+=1
        break
for i in xrange(1000):
    if count[i][0]==0:
      count[i][3]=0.0
      count[i][4]=0.0
    else:
      count[i][3]=float(count[i][1])/count[i][0]
      count[i][4]=float(count[i][2])/count[i][0]
    print("%d %d %d %d %.3f %.3f" %(i,count[i][0],count[i][1],count[i][2],count[i][3],count[i][4]))
