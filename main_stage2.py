
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
parser.add_argument('--n_epoches_1',type=int,default= 200)
parser.add_argument('--batch_size',type=int,default= 512)
parser.add_argument('--species',type=int,default= 1)
parser.add_argument('--pretrain_num',type=int,default= 45)#21,49

opt=vars(parser.parse_args())
ep = opt['n_epoches_1']
pretrain_num = opt['pretrain_num']
specie_num = opt['species']
use_cuda=True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

    
#--------------pretrain g and h for step 1---------------------------------
species_all = ['Homo_sapiens','Rattus_norvegicus','Schistosoma_japonicum','Saccharomyces_cerevisiae','Mus_musculus','Escherichia_coli','Bacillus_velezensis','Plasmodium_falciparum','Oryza_sativa','Arabidopsis_thaliana']

species_list = []

kd_weight = 0.1
for i in range(10):
    if i!= specie_num:
        species_list.append(species_all[i])

# Define the path for the overall result directory
result_path = 'result/'
if not os.path.exists(result_path):
        os.makedirs(result_path)  
# Define the path for the main folder related to the specie_num-th species   
mainfolder = 'result/' + species_all[specie_num] + '/' 

# Define the path for the figure, prediction, feature, model folder within the main folder
if not os.path.exists(mainfolder):
    os.makedirs(mainfolder)
figfolder = mainfolder + '/figure/'
if not os.path.exists(figfolder):
    os.makedirs(figfolder)
prefolder = mainfolder + '/prediction/'
if not os.path.exists(prefolder):
    os.makedirs(prefolder)
feafolder = mainfolder + '/feature/'
if not os.path.exists(feafolder):
    os.makedirs(feafolder)
modelfolder = mainfolder + '/model/'
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)
# Define the path for the data directory
data_path = 'data_ace/data_npy/'
batch_size=opt['batch_size']
# Create a test and valid dataset object for the specie_num-th and general species using the dataset_stage1 class and 
# Create a train dataset object for the specie_num-th and other all species using the dataset_stage1 class(reflecting multi-domain) and 
# Create a data loader for the these dataset with the specified batch size and shuffle option
test_CRC_data1 = dataset_stage1(data_path,'general',species_all[specie_num], mode = 'test')
test_data_loader1 = torch.utils.data.DataLoader(test_CRC_data1, batch_size=batch_size, shuffle=True) 

train_CRC_data1 = dataset_stage2(data_path,species_list,species_all[specie_num], mode = 'train')
train_data_loader1 = torch.utils.data.DataLoader(train_CRC_data1, batch_size=batch_size, shuffle=True) 

validation_CRC_data1 = dataset_stage1(data_path,'general',species_all[specie_num], mode = 'valid')
validation_data_loader1 = torch.utils.data.DataLoader(validation_CRC_data1, batch_size=batch_size, shuffle=True) 

# Function to plot the loss over iterations
def plot_loss(y):

    x = range(0,len(y))
    plt.plot(x, y, '.-',color="red")
    plt_title = 'xxx'
    plt.title(plt_title)
    plt.xlabel('per 200 times')
    plt.ylabel('LOSS')		
    plt.savefig(mainfolder+'ptm-{:s}-loss.png'.format('HUMAN'))

# Function to save the feature matrix as a text file    
def save_feature(feature,phase,index):
    l = feature.shape[1]
    num = feature.shape[0]
    final_matrix = np.ones((num,l))
    for i in range(num):
        temp = feature[i].flatten()
        final_matrix[i] = temp
    np.savetxt(feafolder + '/feature_{:s}_{:d}.txt'.format(phase,index),final_matrix)
    
# Create instances of the classifier for all species and  an encoder model from the densenet module      
classifier0=densenet.Classifier0()
classifier1=densenet.Classifier1()
classifier2=densenet.Classifier2()
classifier3=densenet.Classifier3()
classifier4=densenet.Classifier4()
classifier5=densenet.Classifier5()
classifier6=densenet.Classifier6()
classifier7=densenet.Classifier7()
classifier8=densenet.Classifier8()
classifier9=densenet.Classifier9()
encoder=densenet.Encoder()

print('net--------------------------')
print(encoder)

# Define the loss function as cross-entropy loss
loss_fn=torch.nn.CrossEntropyLoss().cuda()
# Define the loss function as kdloss loss
kdloss = KDLoss(1).cuda()
# Function to calculate the Euclidean distance 
def euclidean_distance(y_true, vects):
    eps = 1e-08 
    margin = 1
    x, y = vects
    temp = torch.sum(torch.square((x - y)/2.0), axis=1, keepdims=True)
    eps_tensor = eps*torch.ones_like(temp)
    zero = torch.zeros(0)
    y_pred = torch.sqrt(torch.maximum(temp, eps_tensor))
    out = torch.mean(y_true * torch.square(y_pred)/torch.pow(torch.max(y_pred),2) + (1 - y_true) * torch.square(torch.maximum(margin - y_pred/torch.pow(torch.max(y_pred),2), 0*torch.ones_like(y_pred))))
    return out
# Load the pre-trained encoder and classifier model using the Rattus_norvegicus species to pretrain

encoder = torch.load('result/pretrain/Homo_sapiens/model/encoder_stage1_{:d}.pth.tar'.format(pretrain_num))   
classifier0 = torch.load('result/pretrain/Homo_sapiens/model/classifier_stage1_{:d}.pth.tar'.format(pretrain_num))  

optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier0.parameters()),lr=0.0005,weight_decay=0.0001)
Loss = []
auc_all=[]
encoder.to(device)

classifier0.to(device)
classifier1.to(device)
classifier2.to(device)
classifier3.to(device)
classifier4.to(device)
classifier5.to(device)
classifier6.to(device)
classifier7.to(device)
classifier8.to(device)
classifier9.to(device)

# Main training loop for the specified number of epochs
for epoch in range(opt['n_epoches_1']):
    print(epoch)
    
    
    for i, (data, labels) in enumerate(train_data_loader1):
        #the 0 stands for the target domain data while the 1-9 stand for the other domains
        data0 = data[0] #target domain data
        data1 = data[1]
        data2 = data[2]
        data3 = data[3]
        data4 = data[4]
        data5 = data[5]
        data6 = data[6]
        data7 = data[7]
        data8 = data[8]
        data9 = data[9]        
        
        labels0 = labels[0] #target domain label
        labels1 = labels[1]
        labels2 = labels[2]
        labels3 = labels[3]
        labels4 = labels[4]
        labels5 = labels[5]
        labels6 = labels[6]
        labels7 = labels[7]
        labels8 = labels[8]
        labels9 = labels[9]

        labels0 = labels0.to(device) 
        labels1 = labels1.to(device)        
        labels2 = labels2.to(device)
        labels3 = labels3.to(device)
        labels4 = labels4.to(device)        
        labels5 = labels5.to(device)
        labels6 = labels6.to(device)        
        labels7 = labels7.to(device)        
        labels8 = labels8.to(device)
        labels9 = labels9.to(device)
 
        labels0 = Variable(labels0)
        labels1 = Variable(labels1)
        labels2 = Variable(labels2)
        labels3 = Variable(labels3)
        labels4 = Variable(labels4)
        labels5 = Variable(labels5)
        labels6 = Variable(labels6)
        labels7 = Variable(labels7)
        labels8 = Variable(labels8)
        labels9 = Variable(labels9)

        data0 = data0.to(device) 
        data1 = data1.to(device)        
        data2 = data2.to(device)
        data3 = data3.to(device)
        data4 = data4.to(device)        
        data5 = data5.to(device)
        data6 = data6.to(device)        
        data7 = data7.to(device)        
        data8 = data8.to(device)
        data9 = data9.to(device)

        data0 = Variable(data0,requires_grad=True)
        data1 = Variable(data1,requires_grad=True)
        data2 = Variable(data2,requires_grad=True)
        data3 = Variable(data3,requires_grad=True)
        data4 = Variable(data4,requires_grad=True)
        data5 = Variable(data5,requires_grad=True)
        data6 = Variable(data6,requires_grad=True)
        data7 = Variable(data7,requires_grad=True)
        data8 = Variable(data8,requires_grad=True)
        data9 = Variable(data9,requires_grad=True)

        optimizer.zero_grad()
        #extracting the output of encoder of all species
        feature0 =encoder(data0)
        y_pred0 = classifier0(feature0)

        feature1 =encoder(data1)
        y_pred1 = classifier1(feature1)

        feature2 =encoder(data2)
        y_pred2 = classifier2(feature2)

        feature3 =encoder(data3)
        y_pred3 = classifier3(feature3)

        feature4 =encoder(data4)
        y_pred4 = classifier4(feature4)

        feature5 =encoder(data5)
        y_pred5 = classifier5(feature5)

        feature6 =encoder(data6)
        y_pred6 = classifier6(feature6)

        feature7 =encoder(data7)
        y_pred7 = classifier7(feature7)

        feature8 =encoder(data8)
        y_pred8 = classifier8(feature8)
        
        feature9 =encoder(data9)
        y_pred9 = classifier9(feature9)
        #calculate the cross-entropy loss for every species between the predicted value and the true label
        loss0 = loss_fn(y_pred0,labels0)
        loss1 = loss_fn(y_pred1,labels1)
        loss2 = loss_fn(y_pred2,labels2)
        loss3 = loss_fn(y_pred3,labels3)
        loss4 = loss_fn(y_pred4,labels4)
        loss5 = loss_fn(y_pred5,labels5)
        loss6 = loss_fn(y_pred6,labels6)
        loss7 = loss_fn(y_pred7,labels7)
        loss8 = loss_fn(y_pred8,labels8)
        loss9 = loss_fn(y_pred9,labels9)
         #calculate the kdloss between the target species and every other species
        kd_loss1 = kdloss(10*feature0, 10*feature1)
        kd_loss2 = kdloss(10*feature0, 10*feature2)
        kd_loss3 = kdloss(10*feature0, 10*feature3)
        kd_loss4 = kdloss(10*feature0, 10*feature4)
        kd_loss5 = kdloss(10*feature0, 10*feature5)
        kd_loss6 = kdloss(10*feature0, 10*feature6)
        kd_loss7 = kdloss(10*feature0, 10*feature7)
        kd_loss8 = kdloss(10*feature0, 10*feature8)
        kd_loss9 = kdloss(10*feature0, 10*feature9) 
        #calculate the total kdloss 
        kd_all = 1/kd_loss1 + 1/kd_loss2 + 1/kd_loss3 + 1/kd_loss4 + 1/kd_loss5 + 1/kd_loss6 + 1/kd_loss7 + 1/kd_loss8 + 1/kd_loss9
        '''
        # all the species use the same weight
        kd_w1 = 1/9   
        kd_w2 = 1/9  
        kd_w3 = 1/9  
        kd_w4 = 1/9  
        kd_w5 = 1/9  
        kd_w6 = 1/9  
        kd_w7 = 1/9 
        kd_w8 = 1/9 
        kd_w9 = 1/9 
        '''
        
        # the weight of each species is calculated by their kdloss
        kd_w1 =  1/kd_loss1/kd_all   
        kd_w2 =  1/kd_loss2/kd_all 
        kd_w3 = 1/kd_loss3/kd_all 
        kd_w4 = 1/kd_loss4/kd_all 
        kd_w5 = 1/kd_loss5/kd_all 
        kd_w6 = 1/kd_loss6/kd_all 
        kd_w7 = 1/kd_loss7/kd_all 
        kd_w8 = 1/kd_loss8/kd_all 
        kd_w9 = 1/kd_loss9/kd_all 
        
        #calculate the total loss for each species
        L1 = loss1 + kd_weight *kd_loss1
        L2 = loss2 + kd_weight *kd_loss2
        L3 = loss3 + kd_weight *kd_loss3
        L4 = loss4 + kd_weight *kd_loss4
        L5 = loss5 + kd_weight *kd_loss5
        L6 = loss6 + kd_weight *kd_loss6
        L7 = loss7 + kd_weight *kd_loss7
        L8 = loss8 + kd_weight *kd_loss8
        L9 = loss9 + kd_weight *kd_loss9

        #calculate the total loss
        loss = loss0 + 0.3*(kd_w1*L1 + kd_w2*L2 + kd_w3*L3 + kd_w4*L4 + kd_w5*L5 + kd_w6*L6 + kd_w7*L7 + kd_w8*L8 + kd_w9*L9)
        loss.backward()
        optimizer.step()

        # Concatenate the predicted outputs and labels for all batches in the validation set(specie_num-th species)

        if i==(len(train_data_loader1)-1):    #i%20==0 or 
            acc=0
            auc = 0
            y_test_pred_all = 0
            labels_all = 0
            for i, (data, labels) in enumerate(validation_data_loader1):
        
                data1 = data
                data1=data1.to(device)
                labels1 = labels
                labels1=labels1.to(device)
                #labels1 = labels1.long()
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
                
                labels1 = labels1.detach().cpu()
                labels1 = labels1[:,1]
                acc+=(torch.max(y_test_pred.detach().cpu(),1)[1]==labels1).float().mean().item()

            labels_all = labels_all[:,1]
            
            accuracy=round(acc / float(i+1), 3)
            # Calculate the AUC (Area Under the Curve) for the validation set
            auc = metrics.roc_auc_score(labels_all,y_test_pred_all[:, 1])
            #print("step1----Epoch %d/%d  valid accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
            #print("step1----Epoch %d/%d  valid auc : %.3f "%(epoch+1,opt['n_epoches_1'],auc))
    
        
            acc=0
            auc = 0
            y_test_pred_all = 0
            labels_all = 0
            Loss_temp = []
            for i, (data, labels) in enumerate(test_data_loader1):
        
                data1 = data
                data1=data1.to(device)
                labels1 = labels

                labels1=labels1.to(device)
                #labels1 = labels1.long()
                labels1 = Variable(labels1)
                y_test_pred=classifier0(encoder(data1))
                loss = loss_fn(y_test_pred,labels1)
                running_loss = loss.item()
                Loss_temp.append(running_loss)
                labels1 = labels
                if i == 0:
                    y_test_pred_all = y_test_pred.detach().cpu()
                    labels_all = labels1.detach().cpu()
                else:
                    y_test_pred_all = torch.cat((y_test_pred_all,y_test_pred.detach().cpu()), 0)
                    labels_all = torch.cat((labels_all,labels1.detach().cpu()), 0)        
                labels1 = labels1.detach().cpu()
                labels1 = labels1[:,1]
                acc+=(torch.max(y_test_pred.detach().cpu(),1)[1]==labels1.detach().cpu()).float().mean().item()
            loss_mean =  sum(Loss_temp)/len(Loss_temp)   
            Loss.append(loss_mean)
            plot_loss(Loss)
            accuracy=round(acc / float(i+1), 3)
            labels_all = labels_all[:,1]
            auc = metrics.roc_auc_score(labels_all,y_test_pred_all[:, 1])
            #print("step1----Epoch %d/%d  test accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
            print("step1----Epoch %d/%d  test auc : %.3f "%(epoch+1,opt['n_epoches_1'],auc))
            auc_all.append(auc)
            np.savetxt(mainfolder+'auc.txt',np.array(auc_all))
            # Create a figure and axis object for plotting and plot the ROC curve
            fpr_t, tpr_t, _ = metrics.roc_curve(labels_all,y_test_pred_all[:, 1])
            fig, ax = plt.subplots()
            ax.plot(fpr_t, tpr_t, 'b-', label='CNN-test {:.3%}'.format(auc))
            ax.legend(loc='lower right', shadow=True)
            plt.title('test of {:s}'.format(species_all[specie_num]))
            plt.savefig(figfolder+'ptm-{:s}-epoch-{:d}-test_stage1.png'.format(species_all[specie_num], epoch+1))
            plt.close()
            #save the model
            torch.save(encoder, modelfolder + 'encoder_stage1_{:d}.pth.tar'.format(epoch)) 
            torch.save(classifier0, modelfolder + 'classifier_stage1_{:d}.pth.tar'.format(epoch)) 
            #save the prediction scores and labels of the epoch-th epoch
            np.savetxt(prefolder+ '/prediction scores of epoch-{:d}.txt'.format(epoch),y_test_pred_all)
            np.savetxt(prefolder+ '/label of epoch-{:d}.txt'.format(epoch),labels_all)




  
