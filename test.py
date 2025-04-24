
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import torch
from torch.autograd import Variable

import numpy as np
import datasets
from datasets import dataset_stage1,dataset_stage2
import sklearn
from sklearn import metrics
import os
from model import densenet
from model.densenet import KDLoss
from model.BasicModule import BasicModule

os.environ['CUDA_VISIBLE_DEVICES']='0'
#Parse to get the model parameters
parser=argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default= 512)
parser.add_argument('--species2',type=int,default= 1)

opt=vars(parser.parse_args())
use_cuda=True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

    
#--------------pretrain g and h for step 1---------------------------------
#species_all = ['HUMAN','MOUSE','RAT','YEAST','ARATH','TOXGV','EMENI','ORYSJ']
species_all = ['Homo_sapiens','Rattus_norvegicus','Schistosoma_japonicum','Saccharomyces_cerevisiae','Mus_musculus','Escherichia_coli','Bacillus_velezensis','Plasmodium_falciparum','Oryza_sativa','Arabidopsis_thaliana']
specie_num = 6
species_list = []

# Define the path for the overall result directory
kd_weight = 0.1
for i in range(10):
    if i!= specie_num:
        species_list.append(species_all[i])
data_path = 'data_ace/data_npy/'
batch_size=opt['batch_size']
test_CRC_data1 = dataset_stage1(data_path,'general',species_all[specie_num], mode = 'test')
test_data_loader1 = torch.utils.data.DataLoader(test_CRC_data1, batch_size=batch_size, shuffle=True) 


    
# Create instances of the classifier for all species and  an encoder model from the densenet module      
classifier0=densenet.Classifier0()
encoder=densenet.Encoder()

classifier0.to(device)
encoder.to(device)

encoder = torch.load('result/{:s}/model/encoder_stage1.pth.tar'.format(species_all[specie_num]))   
classifier0 = torch.load('result/{:s}//model/classifier_stage1.pth.tar'.format(species_all[specie_num]))

classifier0.to(device)
encoder.to(device)
classifier0.eval()
encoder.eval()

loss_fn=torch.nn.CrossEntropyLoss()

kdloss = KDLoss(1).cuda()
# Function to calculate the Euclidean distance 
def euclidean_distance(y_true, vects):
    eps = 1e-08 
    margin = 1
    x, y = vects
#     y_pred = torch.sqrt(torch.maximum(torch.sum(torch.square(x - y), axis=1, keepdims=True), eps_tensor))
    temp = torch.sum(torch.square((x - y)/2.0), axis=1, keepdims=True)
    eps_tensor = eps*torch.ones_like(temp)
    zero = torch.zeros(0)
    y_pred = torch.sqrt(torch.maximum(temp, eps_tensor))
    #out = torch.mean(y_true * torch.square(y_pred) + (1 - y_true) * torch.square(torch.maximum(margin - y_pred, 0*torch.ones_like(y_pred))))/(torch.pow(torch.max(margin - y_pred),2)*10)
    out = torch.mean(y_true * torch.square(y_pred)/torch.pow(torch.max(y_pred),2) + (1 - y_true) * torch.square(torch.maximum(margin - y_pred/torch.pow(torch.max(y_pred),2), 0*torch.ones_like(y_pred))))
    return out


optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier0.parameters()),lr=0.0005,weight_decay=0.0001)
Loss = []
auc_all=[]
encoder.to(device)
acc=0
auc=0
for i, (data, labels) in enumerate(test_data_loader1):
    data1 = data
    data1=data1.to(device)
    labels1 = labels

    labels1=labels1.to(device)
    labels1 = labels1.long()
    labels1 = Variable(labels1)
    y_test_pred=classifier0(encoder(data1))
    loss = loss_fn(y_test_pred,labels1)
    running_loss = loss.item()
    labels1 = labels
    if i == 0:
        y_test_pred_all = y_test_pred.detach().cpu()
        labels_all = labels1.detach().cpu()
    else:
         y_test_pred_all = torch.cat((y_test_pred_all,y_test_pred.detach().cpu()), 0)
         labels_all = torch.cat((labels_all,labels1.detach().cpu()), 0)        

    acc+=(torch.max(y_test_pred.detach().cpu(),1)[1]==labels1.detach().cpu()).float().mean().item()
accuracy=round(acc / float(i+1), 3)
auc = metrics.roc_auc_score(labels_all,y_test_pred_all[:, 1])
print("test auc : %.3f "%(auc))
