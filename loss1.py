#--coding:utf-8--
import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from utils import softmax_loss
from cfg import cfg

'''
args.sent_len = 90
args.class_num = 19
args.pos_dim = 90
args.mPos = 2.5
args.mNeg = 0.5
args.gamma = 0.05
'''
class RankingLossFunc(nn.Module):
    def __init__(self):
        super(RankingLossFunc, self).__init__()
        #self.mPos = args.mPos
        #self.mNeg = args.mNeg
        #self.gamma = args.gamma
        #self.cuda = args.cuda
        
        self.mPos = 1.0
        self.mNeg = 0.5
        self.mNeg2=0.5
        self.gamma = 2
        self.sep=1e-7
        self.softmax=nn.Softmax()
        #self.cuda = args.cuda

    def forward(self, logit, target):
        ##logit:[batch size, number class]
        ##target:[batch size, class]
	#logit=softmax_loss(logit,dim=1)
        #logit=nn.functional.softmax(logit,dim=1)
        val, ind = torch.topk(logit,2,dim=1) # top2 score
        noneOtherInd = target!=0 # not Other index
        rows = list(range(len(logit))) #row index
        part1 = logit[rows,target] #得到每个句子真是的类别标签得分
#         if torch.min(part1) < -20:
#             part1 = self.gamma*(self.mPos-part1)
#         else:
        
        #inds=(ind[:,0]==target).long()
        #print((ind[:,0]==target).shape,val.shape,inds)
        inds =(ind[:,0]==target).long()##负样本，即为预测错误的样本
        predF = val[rows,inds]##正样本，即为预测正确的样本
        ###print(logit.shape,target.shape,target)
	#div=torch.sum(torch.exp(logit),1)
        
        #predF2=val[:,-1][ind[]]

        part1 = torch.log(1+torch.exp(self.gamma*(self.mPos-part1))) + torch.log(1+torch.exp(self.gamma*(-self.mNeg+predF)))
        #part1 = -torch.log(self.mPos-torch.exp(part1)/div)-torch.log(-self.mNeg+torch.exp(predF)/div) # positive loss
        loss=torch.sum(part1)
        del part1,predF,logit

        
        return loss
    
    
class KL_LossFunc(nn.Module):
    def __init__(self,weights):
        super(KL_LossFunc,self).__init__()
        self.weights=weights
        self.base_classes=cfg.base_classes
        self.same_classes=cfg.same_classes
        print("===losss====",self.same_classes)
        self.mean_std=1.0
        same_ids=[]##根据相似类别的分组得到每个类别对应在base类别中的id编号
        for line in self.same_classes:
            n=[]       
            
            for j in line:
                if j in self.base_classes:
                    n.append(self.base_classes.index(j))
            if len(n)>0:
               same_ids.append(n)
        self.same_ids=same_ids
        print(same_ids)
        
        
    def forward(self,meta1):        
        loss=0.0
        meta1=torch.cat(meta1,0)#[2,15*4,1024,1,1]
        batch=meta1.shape[0]//len(self.base_classes)
        channels=meta1.shape[1]
        meta1=meta1.squeeze(-1).squeeze(-1).view(batch,len(self.base_classes),channels).permute(1,0,2)#shape:[15,4,1024]
        meta1_mean=meta1.mean(1)
        meta1_std=meta1.std(1,unbiased=0.001)
        meta_channels=meta1[0].shape[1]
        #batch=meta1[0].shape[0]//len(self.base_classes)
        ##new sub loss to 
        channel_meta=torch.zeros((len(self.same_ids),channels))
        
        ####
        meta1_sub_mean=meta1_mean.mean(1).view(len(self.base_classes))
        meta1_sub_std=meta1_mean.std(1).view(len(self.base_classes))
        mean_sum=torch.zeros(len(self.same_ids))
        std_sum=torch.zeros(len(self.same_ids))
        mean=torch.zeros(len(self.same_ids))
        std_std=torch.zeros(len(self.same_ids))
        single_std_sum=torch.zeros(len(self.same_ids))
        flag=0
        for line in self.same_ids:
            mean_sub=[]
            std_sub=[]
            mean1=[]
            
            
            meta_std=torch.zeros(len(line))##initialize meta features for the similar classes
            meta_mean=torch.zeros(len(line))
            #meta_std_sum=torch.zeros(len(line))
            for j in range(len(line)):
                meta_std[j]=meta1_sub_std.data[line[j]]
                meta_mean[j]=meta1_sub_mean.data[line[j]]
                
            #mean_sum[flag]#torch.exp(meta_sub.std())/torch.exp(meta_sub.std()+self.mean_std*meta_mean.std())###计算相似类别之间的均值分布差  
            #print(meta_std)
            if len(line)==1:
                std_std[flag]=meta1_sub_std.data[line[0]]##相似类别之间的方差的方差
                std_sum[flag]=meta1_sub_std.data[line[0]]#0.0001#torch.FloatTensor([0.00001])
                mean_sum[flag]=meta1_sub_mean.data[line[0]]##相类类别之间的均值的方差
                mean[flag]=meta1_sub_mean.data[line[0]]##相似类别之间的均值的均值
            else:
              
                std_std[flag]=meta_std.std()##相似类别之间的方差的方差
                mean_sum[flag]=meta_mean.std()##相类类别之间的均值的方差
                mean[flag]=meta_mean.mean()##相似类别之间的均值的均值
                std_sum[flag]=torch.sum(meta_std)
            single_std_sum=meta_std.sum()
            flag+=1
            del meta_std,meta_mean
        std_std=Variable(std_std)
        mean_sum=Variable(mean_sum)
        std_sum=Variable(std_sum)
        mean=Variable(mean)     
        #meta1_sum_std_div=torch.exp(torch.sum(meta1_sub_std)).cpu() 
        #print("==========",torch.exp(torch.sum(meta1_sub_std))) 
        #meta1_sub_std=Variable(meta1_sub_std) 
        #index_ids=Variable(torch.LongTensor(self.same_ids),requires_grad=False)          
        #single_std_sum=Variable(torch.FloatTensor(single_std_sum))                   
        for i in range(len(self.same_ids)):
            subloss=Variable(torch.FloatTensor([0.00005]))
            for j in range(i+1,len(self.same_ids)):
                #print("======222=====",torch.log((std_sum[i]-std_sum[j])**2),std_sum[i],std_sum[j],i,j)
                subloss+=torch.exp((std_std[i]-std_std[j])**2)+self.mean_std*torch.exp((mean[i]-mean[j])**2)
                #subloss+=torch.exp((mean_sum[i]-mean_sum[j])**2)+torch.exp((mean[i]-mean[j])**2)
                 
            #loss+=torch.log(1.0+mean_sum[i]/(subloss+mean_sum[i])+std_sum[i])#-torch.log(subloss)
            #index_ids=Variable(torch.LongTensor(self.same_ids[i]).cuda(),requires_grad=False)
            #print("========4444444444=========",torch.sum(torch.gather(meta1_sub_std,dim=0,index=index_ids)))
            m1=meta1_mean[self.same_ids[i]].mean(0)
            loss+=torch.log(1.0+mean_sum[i]/(subloss+mean_sum[i]))#+\
            #self.weights[i]*torch.log(1.0+1./m1.std(0,unbiased=0.001)).cpu()
                    #self.weights[i]*(1.0+\
                    #torch.exp(torch.sum(torch.gather(meta1_sub_std,dim=0,index=index_ids)))/(1.0+\
                    #meta1_sum_std_div))
        del meta1,std_std,mean,meta1_std,meta1_mean,mean_sum#,single_std_sum
        return loss
