
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
from model.BasicModule import BasicModule

os.environ['CUDA_VISIBLE_DEVICES']='0'
#Parse to get the model parameters
parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches_1',type=int,default= 500)
parser.add_argument('--batch_size',type=int,default= 512)


opt=vars(parser.parse_args())
ep = opt['n_epoches_1']
#Configure the model's runtime environment to run on either GPU or CPU, and set a random seed to make the code reproducible.
use_cuda=True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

    
#--------------pretrain g and h for step 1---------------------------------
#List of all species and select the first and second species for the experiment
species_all = ['Homo_sapiens','Rattus_norvegicus','Schistosoma_japonicum','Saccharomyces_cerevisiae','Mus_musculus','Escherichia_coli','Bacillus_velezensis','Plasmodium_falciparum','Oryza_sativa','Arabidopsis_thaliana']
specie1 = species_all[0]
specie2 = species_all[0]
# Define the path for the overall result directory for pretraining
result_path = 'result/pretrain/'
if not os.path.exists(result_path):
        os.makedirs(result_path)
        
# Define the path for the figure, prediction, feature, model folder within the main folder       
mainfolder = 'result/pretrain/' + specie2 + '/' 
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
test_CRC_data1 = dataset_stage1(data_path,specie1,specie2, mode = 'test')
test_data_loader1 = torch.utils.data.DataLoader(test_CRC_data1, batch_size=batch_size, shuffle=True) 
train_CRC_data1 = dataset_stage1(data_path,specie1,specie2, mode = 'train')
train_data_loader1 = torch.utils.data.DataLoader(train_CRC_data1, batch_size=batch_size, shuffle=True) 
validation_CRC_data1 = dataset_stage1(data_path,specie1,specie2, mode = 'valid')
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
    
# Create an instance of the classifier and encoder model from the densenet module        
classifier=densenet.Classifier0()
encoder=densenet.Encoder()

#print('net--------------------------')
#print(encoder)

classifier.to(device)
encoder.to(device)
# Define the loss function as cross-entropy loss
loss_fn=torch.nn.CrossEntropyLoss()
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

# Create an Adam optimizer for the encoder and classifier parameters

optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.0001,betas=(0.9,0.999), eps=1e-08, weight_decay=0.0001)

Loss = []
# Main training loop for the specified number of epochs
for epoch in range(opt['n_epoches_1']):
    optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.0002,weight_decay=0.0001) #学习率参数可调
    for i, (data, labels) in enumerate(train_data_loader1):
        
        data1 = data
        data1=data1.to(device)
        labels1 = labels
        labels1=labels1.to(device)
        data1 = Variable(data1,requires_grad=True)
        labels1 = Variable(labels1)  
        optimizer.zero_grad()
        
        y_pred1 =classifier(encoder(data1))
        loss = loss_fn(y_pred1,labels1)
        loss.backward()
        optimizer.step()
        
    #validation
    
    acc=0
    auc = 0
    y_test_pred_all = 0
    labels_all = 0
    for i, (data, labels) in enumerate(validation_data_loader1):
        
        data1 = data
        data1=data1.to(device)
        labels1 = labels
        labels1=labels1.to(device)
        labels1 = Variable(labels1)
        y_test_pred=classifier(encoder(data1))
        loss = loss_fn(y_test_pred,labels1)
        running_loss = loss.item()
        Loss.append(running_loss)
        # Concatenate the predicted outputs and labels for all batches in the validation set
        if i == 0:
            y_test_pred_all = y_test_pred.detach().cpu()
            labels_all = labels1.detach().cpu()
        else:
            y_test_pred_all = torch.cat((y_test_pred_all,y_test_pred.detach().cpu()), 0)
            labels_all = torch.cat((labels_all,labels1.detach().cpu()), 0)        
        # Calculate the accuracy for the current batch and accumulate it
        acc+=(torch.max(y_test_pred.detach().cpu(),1)[1]==labels1[:,1].detach().cpu()).float().mean().item()

    plot_loss(Loss)
    accuracy=round(acc / float(i+1), 3)
    # Calculate the AUC (Area Under the Curve) for the validation set
    auc = metrics.roc_auc_score(labels_all[:,1],y_test_pred_all[:, 1])

    # Verify the model on the test set and draw the ROC curve.      
    acc=0
    auc = 0
    y_test_pred_all = 0
    labels_all = 0
    for i, (data, labels) in enumerate(test_data_loader1):
        
        data1 = data
        data1=data1.to(device)

        labels1 = labels
        labels1=labels1.to(device)
        labels1 = Variable(labels1)
        y_test_pred=classifier(encoder(data1))
        
        loss = loss_fn(y_test_pred,labels1)
        running_loss = loss.item()
        Loss.append(running_loss)
        
        if i == 0:
            y_test_pred_all = y_test_pred.detach().cpu()
            labels_all = labels1.detach().cpu()
        else:
            y_test_pred_all = torch.cat((y_test_pred_all,y_test_pred.detach().cpu()), 0)
            labels_all = torch.cat((labels_all,labels1.detach().cpu()), 0)        
        # Calculate the accuracy for the current batch and accumulate it
        acc+=(torch.max(y_test_pred.detach().cpu(),1)[1]==labels1[:,1].detach().cpu()).float().mean().item()
       
    plot_loss(Loss)
    accuracy=round(acc / float(i+1), 3)
    auc = metrics.roc_auc_score(labels_all[:,1],y_test_pred_all[:, 1])
    print("step1----Epoch %d/%d  test accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
    print("step1----Epoch %d/%d  test auc : %.3f "%(epoch+1,opt['n_epoches_1'],auc))
    # Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr_t, tpr_t, _ = metrics.roc_curve(labels_all[:,1],y_test_pred_all[:, 1])
    # Create a figure and axis object for plotting and plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr_t, tpr_t, 'b-', label='CNN-test {:.3%}'.format(auc))
    ax.legend(loc='lower right', shadow=True)
    plt.title('test of {:s}'.format(specie2))
    plt.savefig(figfolder+'ptm-{:s}-epoch-{:d}-test_stage1.png'.format(specie2, epoch+1))
    plt.close()
    #save the model
    torch.save(encoder, modelfolder + 'encoder_stage1_{:d}.pth.tar'.format(epoch)) 
    torch.save(classifier, modelfolder + 'classifier_stage1_{:d}.pth.tar'.format(epoch)) 


