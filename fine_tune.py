# -*- coding: utf-8 -*-s

import time
import copy 

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from torchvision import transforms

from preprocessing import generate_stft_images

class to_dataset(Dataset):
    """
    class to make x and y into a Dataset for pytorch training
        convert x (np.darray of images into tensors, also applying other transforms)
        convert y (np.darray of int, into tensor of type long) 
    """
    
    def __init__(self,x,y,mode):
        
        #if x contains audio, we compute the images
        if ( len(x.shape) <= 3) :
            print("input needs to be images not audio array")
            #print("Generating images~~")
            #images = generate_stft_images(x,toint = True)
            
        #else if x already contains images
        else :
            images = x
            
        self.data = generate_dcnn_input(images,mode)
        self.labels = torch.tensor(y,dtype = torch.long)
        
    def __getitem__(self,index):
        return self.data[index], self.labels[index]
        
        
    def __len__(self):
        return len(self.data)
    

def create_dataloader(x,y,batch_size, mode,shuffle = True, save = False, path = 'dataloader' ):
    """

    Parameters
    ----------
    x : data 
    y : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    shuffle : TYPE, optional
        DESCRIPTION. The default is True.
    save : TYPE, optional
        DESCRIPTION. The default is False.
    path : TYPE, optional
        DESCRIPTION. The default is 'dataloader'.
    mode : string, "train" or other
        determine what transforms we will apply depending on the data
    Returns
    -------
    dataloader : TYPE
        DESCRIPTION.

    """
    print("Creating Dataloader")
    dataset = to_dataset(x,y,mode)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size,shuffle= shuffle)
    
    if save:
        print("Saving Dataloader~~")
        torch.save(dataloader,path)
        print("Dataloader saved as",path)
    
    return dataloader


def normalize_with_mean_std(images,mean,std):
    """
    normalize a set of images among the 3 channels RGB given mean and std

    Parameters
    ----------
    image : array of images (n_images,227,227,3)
    mean : array of float (3,)
    std : array of float (3,)

    Returns
    -------
    normalized : the normalized set of images

    """
    normalized = images
    for i in range(len(mean)):
        normalized[:,:,:,i] = (images[:,:,:,i] - mean[i]) / std[i]
        
    return normalized


def generate_single_input(image,mode) : 
    """
    basically convert an image of type np.darray to tensor, apply some transforms sometimes ...
    
    Parameters
    ----------
    image : (227,227,3)

    Returns
    -------
    tens : tensor of the image ()

    """
    ##apply transform for training datas
    if mode == "train" :
        transform = transforms.Compose([
            transforms.ToTensor(), #normalize [0 1] and change into tensor [3,227,227]
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #Random_shift(n_shift = 5)
            #Gaussian_Noise(mean = 0,std = 0.3)
            #transforms.RandomApply([Gaussian_Noise(mean = 0,std = 0.3)],p=0.5)
            ])
        tens = transform(image)
    
    #apply transform for non trianing data
    else :
        transform = transforms.Compose([
            transforms.ToTensor(), #normalize [0 1] and change into tensor [3,227,227]
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        tens = transform(image)
    
    if (len(tens.size())<4) :
        tens = torch.unsqueeze(tens, 0) #for the batch_size dimension [1,3,227,227]
    
    return tens


def generate_dcnn_input(images,mode):
    """
    Converts images into tensors by applying transforms an array of images 
    in order to make it usable by alexnet network.
    calling generate_signe_input() for image in images
    
    Parameters
    ----------
    images : np.darray of RGB images (nb_images,227,227,3) 

    Returns
    -------
    input_tensor : dtype TENSOR (nb_images,3,227,227)

    """

    #init an empty tensor
    input_tensor = torch.empty(size = (images.shape[0],images.shape[-1],images.shape[1],images.shape[2]))
    
    #test = torch.empty(size = (images.shape[0]*6,images.shape[-1],images.shape[1],images.shape[2]))
    
    #for each image, apply the transform
    for i, image in enumerate(images) :
        temp = generate_single_input(image,mode)
        input_tensor[i,:,:,:] = temp
        
        #test[i*5:(i+1)*5,:,:,:] = temp
        
    return input_tensor


def freeze_layers(model,freeze_features = True,freeze_classifier = False):
    """
    set requires_grad for false depending on the layers
    ----------
    model : used DCNN model
    freeze_features : bool
        if we want to freeze all the features layers
    freeze_classifier : bool
        if we want to freeze the 2 first layers (3 in total, last with number of classes)
    """

    if freeze_features :
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    
    if freeze_classifier :
        for i, p in enumerate(model.classifier.parameters()):
            if i < 2 :

                p.requires_grad = False
    
    print("Model frozen ~")


def is_frozen(model,show_param = True):
    
    for p in model.parameters():
        print(p.requires_grad, end =" ")
        if show_param : 
            print(p.shape)


def create_alexnet(n_classes=1000,pretrained = True):
    """
    create alexnet network
    Parameters
    ----------
    n_classes : int
        Number of classes considered for last FC layer. The default is 1000.
    pretrained : bool
        whether the crated alexnet is pretrained or not
        
    Returns
    -------
    alexnet : model
    """
    
    if pretrained : 
        alexnet = models.alexnet(pretrained = pretrained)
    else: 
        alexnet = models.alexnet()
    
    if n_classes != 1000:
        alexnet.classifier[6] = nn.Linear(4096,n_classes)
        
    return alexnet


def emotions_to_label(dic,emotions_array) : 
    """
    Converts array of emotions into array of label

    Parameters
    ----------
    dic : dictionnary of emotions (unique occurence)
        
    emotions_array : array of strings containing emotions

    Returns
    -------
    label : array of int, containing the index of ith the element of emotions_array in dic
    """

    label = np.array([],dtype=int)
    for emotion in emotions_array :
        index = dic.index(emotion,0,len(dic))
        label = np.append(label,index)
    return label


def train_model(model,dataloaders,criterion,optimizer,num_epochs=25):
    """
    Start training the model

    Parameters
    ----------
    model : alexnet model used
    dataloaders : DataLoaders
        dictionnary of dataloaders : dataloaders = {"train" : dataloader_train, "val" : dataloader_val}
    criterion : criterion used (loss function)
    optimizer : optimizer used (SGD,Adam...)
    num_epochs : int
        Number of epochs for training. The default is 25.

    Returns
    -------
    model : the best model (obtained with the highest accuracy for validation)
    list : [train_acc_history,train_loss_history,val_acc_history,val_loss_history]

    """
    #model.double() ########################

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        model.cuda()
        
    print("device:",device)

    train_acc_history = []
    train_loss_history = []
    
    val_acc_history = []
    val_loss_history = []

    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            
            if phase == 'train':
                #set model to training mode (activate dropout layers etc.)
                model.train() 
                
            else:
                #set model to evaluate mode (deactivate dropout layers etc..)
                #can check with training attribute print(model.dropout_layer.training)
                model.eval() 
                
            running_loss = 0.0
            running_corrects = 0

            for inputs,labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                    #outputs = model(inputs.double()) #########################
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model,[train_acc_history,train_loss_history,val_acc_history,val_loss_history]




