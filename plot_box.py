import numpy as np
import cv2
import os
import math
import torch
import sys
files=sys.argv[1]
splits=files.split('/')
method=splits[1]
novel_id=2#int(splits[2].strip('novel'))-1
if 'shot' in files:
    shot_id=splits[3].split('_')[-2]
else:
    shot_id='1shot'
#files='train_result/both/novel3/metatunetest1_novel2_neg0_metatune_2shot_test/ene000003/'
novels={0:['bird','bus','cow','motorbike','sofa'],
        1:['aeroplane','bottle','cow','horse','sofa'],
        2:['boat','cat','motorbike','sheep','sofa']}
top=10
boxes=[]
threshold=0.0
colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
def get_color(c, x, max_val):
    ratio = float(x)/max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    return int(r*255)
def plot_box_cv2(img,boxes,classes,cls_id,savename,lens=5): 
    width = img.shape[1]
    height = img.shape[0]
    offset = cls_id*123457 % lens
    red   = get_color(2, offset, lens)
    green = get_color(1, offset, lens)
    blue  = get_color(0, offset, lens)
    rgb = (red, green, blue)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(float(box[-4]))
        y1 = int(float(box[-3]))
        x2 = int(float(box[-2]))
        y2 = int(float(box[-1]))
        scores=box[-5]     
        #print(box,img.shape) 
        #img = cv2.putText(img,scores , (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, rgb, 1)
        print(x2-x1,y2-y1,img.shape,savename)
        img = cv2.rectangle(img, (x1-80,y1+140), (x2-100,y2+140), rgb, 1)
    cv2.imwrite(savename, img)
cls_id=0
for classes in novels[novel_id]:
    if 'sofa' not in classes:
        continue
    ff=files+'comp4_det_test_'+classes+'.txt'
    ###fileid,cls_conf,prob,xmin,ymin,xmax,ymax
    f=file(ff,'r')
    contents=f.readlines()
    f.close()
    dicts={}
    for c in contents:
        line=c.strip('\n').split(' ')
        if float(line[-5])<threshold:
            continue
        if line[0] not in dicts:
             dicts[line[0]]=[[line[-5],line[-4],line[-3],line[-2],line[-1]]]
        else:
             dicts[line[0]].append([line[-5],line[-4],line[-3],line[-2],line[-1]])
    #boxes.append(dicts)
    nums=[]
    for d in dicts:
        nums.append(len(dicts[d]))
    print(nums)
    nums=np.argsort(-np.array(nums))
    print(nums)
    final={}
    if classes=="motorbike":
        classes="mbike"
    if classes=="aeroplane":
        classes="aero"
    print(nums,dicts)
    for i in range(200):
        if i>len(nums)-1:
            continue
        #print(nums[i],dicts.keys()[nums[i]])
        final[dicts.keys()[nums[i]]]=dicts[dicts.keys()[nums[i]]]
        savename=dicts.keys()[nums[i]]
        imgfile="/home1/liqian/data/voc2007_2012/VOCdevkit/VOC2007/JPEGImages/"+dicts.keys()[nums[i]]+".jpg"
        #imgfile="results/demo/"+dicts.keys()[nums[i]]+".jpg"
        print(imgfile)
        img=cv2.imread(imgfile)
        #savename="/home1/liqian/Fewshot_Detection/plot/atlas_plot/"+splits[2]+"/"+shot_id+"/"+method+"/"+classes+"/"+dicts.keys()[nums[i]]+".jpg"
        savename="/home1/liqian/Fewshot_Detection/plot/atlas_plot/"+classes+"/"+dicts.keys()[nums[i]]+".jpg"
        plot_box_cv2(img,dicts[dicts.keys()[nums[i]]],classes,cls_id,savename,lens=5)
    cls_id+=1    
    boxes.append(final)
