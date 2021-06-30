# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


from fine_tune import generate_dcnn_input, create_dataloader, create_model, emotions_to_label, train_model
from data_augmentation import random_shift_data, audio_augmentation
from preprocessing import generate_stft_images

from sklearn.metrics import confusion_matrix
import seaborn as sns

def f1(accuracy,recall):
    return 2* (accuracy*recall/(accuracy+recall) )
    
def plot_history(h):
    """
    PLOT LOSS AND ACCURACY OF TRAIN AND VALIDATION
    Parameters
    ----------
    h : history of training, list of vector 
    containing [train_acc_history,train_loss_history,val_acc_history,val_loss_history]
    """
    fig1,ax1 = plt.subplots()
    ax1.plot(h[0],'r',label = "train accuracy")
    ax1.plot(h[2],'b',label = "validation accuracy")
    ax1.set_xlabel("Epochs")
    ax1.legend()

    
    fig2,ax2 = plt.subplots()
    ax2.plot(h[1],'r',label = "train loss")
    ax2.plot(h[3],'b',label = "validation loss")
    ax2.set_xlabel("Epochs")
    ax2.legend()
    
def plot_cm(cm,dic_emotions):
    
    
    f, ax = plt.subplots(figsize = (7,7))
    sns.heatmap(np.round(cm*100,decimals = 2), annot=True, fmt='g', ax=ax,cmap="YlGnBu");
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(dic_emotions); 
    ax.yaxis.set_ticklabels(dic_emotions);
        
    
    
def output_to_emotions(y_pred,dic_emotions):
    """
    convert prediction of network into label of emotions -> decoder
    Parameters
    ----------
    y_pred : torch tensor, output of network (nb_exemples, nb_emotions)
    dic_emotions : dictionnary of emotions

    Returns
    -------
    predicted emotions

    """
    pred = []
    for i in range(len(y_pred)):
        int_label = torch.argmax(y_pred[i]).numpy()
        pred.append(dic_emotions[int_label])
    
    return np.array(pred)


def eval_model_get_cm(xtest,ytest,model_name,model_path, dic_emotions, show_cm = False):
    """
    evaluate a model and compute confusion matrix

    Parameters
    ----------
    xtest : images to predict (nb_exemples,227,227,3)
    ytest : label containing emotions 

    model_path : path of the considered model

    show_cm : boolean show computed cm

    Returns
    -------
    C : TYPE
        DESCRIPTION.

    """
    #load model
    n_classes = len(dic_emotions)
    m = create_model(model_name,n_classes,pretrained = True)
    m.load_state_dict(torch.load(model_path))
    m.eval()
    
    #generate model input
    xtest = generate_dcnn_input(xtest,mode='test')
    
    #predict
    ypred = m(xtest)
    ypred = output_to_emotions(ypred,dic_emotions)

    #C = confusion_matrix(y_true = ytest, y_pred = ypred, labels = dic_emotions,normalize = 'pred')
    C = confusion_matrix(y_true = ytest, y_pred = ypred, labels = dic_emotions)
    
    if show_cm :
        plot_cm(C,dic_emotions)
    
    return C


class Training :
    
    def __init__(
            self, 
            dic_emotions, 

            data,  #images
            labels,  #labels
            
            n_epochs,  
            lr, 
            momentum, 
            batch_size, 

            #data augmentation
            rcs=None, 
            noise=False,
            raw_data = None, #audio
 
            ):
        """
        data : array of float images (nb_exemples,227,227,3) for exemple
        labels : labels of string emotions (nb_exemples,)
        lr : float/double
        momentum : float/double
        n_epochs : int
        batch_size : int
        """

        self.dic_emotions = dic_emotions
        

            
        self.data = data #array of images
        self.labels = labels

        self.n_epochs = n_epochs
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

        #data augm
        self.rcs = rcs
        self.noise = noise
        try : 
            self.raw_data = raw_data.copy() #array of object (because audio of different size)
        except : 
            self.raw_data = raw_data
            
        #stft
        self.Fs = None,
        self.signal_len = None,
        self.n_fft = None,
        self.hop_length = None,
        self.win_length = None,
        


    def set_stft(self,Fs,signal_len,n_fft,hop_length,win_length):
        self.Fs = Fs
        self.signal_len = signal_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        

    def start(self,test_index,model_name,model_path,save_model = False, save_path = ""):
        """
        start a training session, generate input for alexnet network (compute images/split data/crete dataloaders)
        create network, optimizer, criterion and start learning process
    
        Parameters
        ----------

    
        test_index : index for validation data, tuple [start index, end index]
        model_name : string, name of pretrained model we want to use, ex : "alexnet"
        model_path : string, path of weights if transfer learning 
        save_model : bolean, save the trained model 
        save_path : string,path of saving

    
        Returns
        -------
        best_model : the model with highest validation accuracy during training session
        history : history of training, list of vector 
        containing [train_acc_history,train_loss_history,val_acc_history,val_loss_history]
    
        """
        if ( self.noise and (self.raw_data is None) ) :
            print("NOISE = TRUE BUT NO AUDIO DATA TO PERFORM AUGMENTATION")
            sys.exit()
            
        print("Starting a training session")
        ###SPLITTING INTO TRAIN AND VALIDATION###

        try : 
            xval = self.data[test_index[0]:test_index[1]]
            xtrain = np.delete(self.data,np.arange(test_index[0],test_index[1]),axis=0)

            yval = emotions_to_label(self.dic_emotions,self.labels[test_index[0]:test_index[1]])
            ytrain = emotions_to_label(self.dic_emotions,np.delete(self.labels,np.arange(test_index[0],test_index[1])))
            index_type = "interval"
            print("WITH INTERVAL OF TEST INDEX")
        except :
            xval = self.data[test_index]
            xtrain = np.delete(self.data,test_index,axis=0)
        
            yval = emotions_to_label(self.dic_emotions,self.labels[test_index])
            ytrain = emotions_to_label(self.dic_emotions,np.delete(self.labels,test_index))
            index_type = "explicit"
            print("WITH EXPLICIT INDEX")
        
        if self.noise :
            
            if index_type == "interval" :
                xa,ya = audio_augmentation(np.delete(self.raw_data,np.arange(test_index[0],test_index[1]),axis=0),ytrain,self.Fs,p=0.5)

            if index_type == "explicit" :
                xa,ya = audio_augmentation(np.delete(self.raw_data,test_index,axis=0),ytrain,self.Fs,p=0.5)
                
            xa = generate_stft_images(xa , out_size = xtrain[0].shape[0:2],
                                      signal_len = self.signal_len,Fs=self.Fs,
                                      n_fft = self.n_fft, hop_length = self.hop_length, 
                                      win_length = self.win_length, save = False, toint = True)
            
            
        if self.rcs != None :
            xtrain,ytrain = random_shift_data(xtrain,ytrain,n_shift=self.rcs,save=False)
            
        if self.noise :
            xtrain = np.append(xtrain,xa,axis=0)
            ytrain = np.append(ytrain,ya,axis=0)
        
        print("train :" ,xtrain.shape, "ytrain :",ytrain.shape)
        print("val :" ,xval.shape, "yval :",yval.shape)

        ###CREATING DATALOADERS###
        dataloader_train = create_dataloader(x=xtrain,y=ytrain, mode = "train",save = False,path = "dataloader_train16_shift",batch_size=self.batch_size)
        dataloader_val = create_dataloader(x=xval,y=yval,mode = "val",save = False,path = "dataloader_val16_shift",batch_size=self.batch_size)
        dataloaders = {"train" : dataloader_train, "val" : dataloader_val}
        
        ###CREATING MODEL###
        model = create_model(model_name=model_name,n_classes=len(self.dic_emotions),pretrained = True)
        print("%s model with %s classes created"%(model_name,len(self.dic_emotions)))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        #optimizer = optim.Adam(alexnet.parameters(),lr=0.0001)

        if model_path != None :
            loaded = torch.load(model_path)
            loaded.pop(list(loaded.keys())[-1])
            loaded.pop(list(loaded.keys())[-1])
            model.load_state_dict(loaded,strict=False)
            print("Loaded weight from %s"%model_path)
            
        ###START TRAINING###
        print("Starting training \n", '-'*20)
        print("n_epochs={}, batch_size={},lr={},momentum={},val_split={}".format(self.n_epochs,self.batch_size,self.lr,self.momentum,test_index))
        best_model, history, time_elapsed = train_model(model,dataloaders,criterion,optimizer,self.n_epochs,inception = (model_name=="inception"))
    
        ###SAVE###
        if save_model : 
            torch.save(best_model.state_dict(),save_path)
            np.save(arr = history,file = "%s_history"%save_path)
            print("SAVED as %s and %s_history"%(save_path,save_path))
    
        return best_model, history, time_elapsed
    
    def eval(self,test_index,model_name,models_path):
        print("EVALUATING MODEL %s AND PLOTTING CONFUSION MATRIX"%models_path)
        xtest = self.data[test_index[0]:test_index[1]]
        ytest = self.labels[test_index[0]:test_index[1]]
        h_path = "%s_history.npy"%models_path
        h = np.load(file = h_path,allow_pickle=True)
        C = eval_model_get_cm(xtest,ytest,model_name,models_path,self.dic_emotions,show_cm = True)
        print(np.sum(np.diagonal(C*100)))
        plot_history(h)
        return C
    
class Losgo_e5(Training):
    
    def __init__(self,
                 n_losgo,
                 dic_emotions,
                 

                 data,
                 label,
                 
                 n_epochs,
                 lr, 
                 momentum, 
                 batch_size, 
                 
                 rcs = None,
                 noise = False,
                 raw_data = None):
        
        super().__init__(dic_emotions,
                         

                         data,label,
                         
                         n_epochs, lr, 
                         momentum, batch_size,
                         
                         rcs,noise,raw_data)
        
        self.n_losgo = n_losgo
    
        self.kfold = self.compute_kfold()
        

    def set_n_losgo(self,n_losgo) :
        self.n_losgo = n_losgo
        self.kfold = self.compute_kfold()
        
    def compute_kfold(self):
        last_emo = 0
        bracket_index = []
        for i,emotion in enumerate(self.labels):
            if emotion != "surprise" and last_emo == "surprise":
                bracket_index.append(i)
                
            last_emo = emotion
        bracket_index.append(len(self.labels))
        
        kfold = []
        for i in range(len(bracket_index)//self.n_losgo):
            if i==0 :
                kfold.append([0,bracket_index[(i+1)*self.n_losgo-1]])
            else : 
                kfold.append([bracket_index[(i)*self.n_losgo-1],bracket_index[(i+1)*self.n_losgo-1]])
        if(len(bracket_index)%self.n_losgo != 0):
            kfold.append([bracket_index[-self.n_losgo],bracket_index[-1]])
        
        return kfold
    
    def start(self,model_name,model_path,save_model = False,save_path= "" ):

        print("INDEXS FOR LOSGO  :\n",self.kfold)

        print("CROSS VALIDATION USING LOSGO METHOD WITH %s SPEAKERS OUT"%self.n_losgo)
        best_vals = []
        for i in range(len(self.kfold)):
            test_index = self.kfold[i]
                
            best_model, history,_ = super().start(test_index=test_index,
            model_name = model_name,
            model_path = model_path,
            save_model=save_model,
            save_path="%sbest_model%s"%(save_path,test_index))
                
            best_vals.append(np.max(history[2]))
                
        return best_vals
        
    def eval(self,model_name,models_path):
        
        print("EVALUATING MODEL %s AND PLOTTING CONFUSION MATRIX"%models_path)
        mean_cm = np.zeros((len(self.dic_emotions),len(self.dic_emotions)),dtype = np.float64)
        
        val_acc = dict()
        for i in range(len(self.kfold)):
            test_index = self.kfold[i]
            #consturciton donnée
            xval = self.data[test_index[0]:test_index[1]]
            yval = self.labels[test_index[0]:test_index[1]]
                
            m_path = "%sbest_model%s"%(models_path,test_index)
            C = eval_model_get_cm(xval,yval,model_name,m_path,self.dic_emotions,show_cm= False)
            mean_cm = mean_cm + C
            
            #historique
            h = np.load("%s_history.npy"%(m_path),allow_pickle = True)
            val_acc[i] = h[2]
            
        #mean_cm = mean_cm/len(self.kfold)
        """average accuracy"""
        for i in range(len(self.dic_emotions)):
            mean_cm[:,i]=mean_cm[:,i]/np.sum(mean_cm[:,i])
        
        """average recall"""
        # for i in range(len(self.dic_emotions)):
        #      mean_cm[i,:]=mean_cm[i,:]/np.sum(mean_cm[i,:])

        plot_cm(mean_cm,self.dic_emotions)
        
        return mean_cm, val_acc


class Loso_edb(Training):
    
    def __init__(self,
                 subject_label,
                 dic_emotions,
                 
                 data,
                 label, 
                 
                 n_epochs, 
                 lr, 
                 momentum, 
                 batch_size, 
                
                 rcs = None ,
                 noise=False, 
                 raw_data = None):
        
        
        super().__init__(dic_emotions,
                         
                         data,label,
                         
                         n_epochs, lr, 
                         momentum, batch_size,
                         
                         rcs,noise,raw_data)
        
        self.subject_label = subject_label
        self.subject = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
        
    def start(self,model_name, model_path, save_model = False,save_path= "" ):

        print("subjects: %s"%self.subject)
        print("CROSS VALIDATION USING LOSO METHOD")
        best_vals = []
        for a in (self.subject) : 
            print("SUBJECT LEFT OUT %s"%a)
            test_index = np.where(self.subject_label == a)
                
            best_model, history,_ = super().start(test_index,
            model_name = model_name,
            model_path = model_path,
            save_model=save_model,
            save_path = "%sLOSO_s%s"%(save_path,a))
            
            best_vals.append(np.max(history[2]))
            
        return best_vals
    
    def eval(self,model_name,models_path):
        print("EVALUATING MODEL %s AND PLOTTING CONFUSION MATRIX"%models_path)
        mean_cm = np.zeros((len(self.dic_emotions),len(self.dic_emotions)),dtype = np.float64)
        
        val_acc = dict()
        for i,a in enumerate(self.subject): 
            test_index = np.where(self.subject_label == a)
            #consturciton donnée
            xtest = self.data[test_index]
            ytest = self.labels[test_index]
                       
            #matrice de confusion
            m_path = "%sLOSO_s%s"%(models_path,a)
            C = eval_model_get_cm(xtest,ytest,model_name,m_path,self.dic_emotions,show_cm= False)                
            mean_cm = mean_cm + C

            #historique
            h = np.load("%s_history.npy"%m_path,allow_pickle = True)
            val_acc[a] = h[2]
            
        #mean_cm = mean_cm/len(self.subject)
        """average accuracy"""
        for i in range(len(self.dic_emotions)):
            mean_cm[:,i]=mean_cm[:,i]/np.sum(mean_cm[:,i])
        
        """average recall"""
        # for i in range(len(self.dic_emotions)):
        #       mean_cm[i,:]=mean_cm[i,:]/np.sum(mean_cm[i,:])

        plot_cm(mean_cm,self.dic_emotions)
        
        return mean_cm, val_acc
    
