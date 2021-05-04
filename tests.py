# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


from fine_tune import generate_dcnn_input, create_dataloader, create_alexnet, emotions_to_label, train_model
from data_augmentation import random_shift_data

from sklearn.metrics import confusion_matrix
import seaborn as sns


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


def eval_model_get_cm(xtest,ytest,model_path, dic_emotions, show_cm = False):
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
    alexnet = create_alexnet(n_classes,pretrained = True)
    alexnet.load_state_dict(torch.load(model_path))
    alexnet.eval()
    
    #generate model input
    xtest = generate_dcnn_input(xtest,mode='test')
    
    #predict
    ypred = alexnet(xtest)
    ypred = output_to_emotions(ypred,dic_emotions)
    C = confusion_matrix(y_true = ytest, y_pred = ypred, labels = dic_emotions,normalize = 'pred')
    
    if show_cm :
        f, ax = plt.subplots(figsize = (7,7))
        sns.heatmap(C*100, annot=True, fmt='g', ax=ax,cmap="YlGnBu");
        ax.xaxis.tick_top()
        ax.xaxis.set_ticklabels(dic_emotions); 
        ax.yaxis.set_ticklabels(dic_emotions);
    
    return C


class Training :
    
    def __init__(self, data, labels, n_epochs, lr, momentum, batch_size, dic_emotions,rcs,noise=None):
        """
        data : consider data, could be array of audio or images (nb_exemples,audio_len) or (nb_exemples,227,227,3)
        labels : labels of emotions (nb_exemples,)
        lr : float/double
        momentum : float/double
        n_epochs : int
        batch_size : int
        """
        try :
            self.data = data.copy()
        except :
            self. data = data
        self.labels = labels
        
        #data augm
        self.rcs = rcs
        self.noise = noise
        
        self.n_epochs = n_epochs
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.dic_emotions = dic_emotions



        
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
        print("Starting a training session")
        try :
        
            xval = self.data[test_index[0]:test_index[1]]
            xtrain = np.delete(self.data,np.arange(test_index[0],test_index[1]),axis=0)

            yval = emotions_to_label(self.dic_emotions,self.labels[test_index[0]:test_index[1]])
            ytrain = emotions_to_label(self.dic_emotions,np.delete(self.labels,np.arange(test_index[0],test_index[1])))
            print("WITH INTERVAL OF TEST INDEX")
        except :
            xval = self.data[test_index]
            xtrain = np.delete(self.data,test_index,axis=0)
        
            yval = emotions_to_label(self.dic_emotions,self.labels[test_index])
            ytrain = emotions_to_label(self.dic_emotions,np.delete(self.labels,test_index))
            print("WITH EXPLICIT INDEX")
        
        if self.rcs > 0 :
            xtrain,ytrain = random_shift_data(xtrain,ytrain,n_shift=self.rcs,save=False)
            
        print("train :" ,xtrain.shape, "ytrain :",ytrain.shape)
        print("val :" ,xval.shape, "yval :",yval.shape)

        dataloader_train = create_dataloader(x=xtrain,y=ytrain, mode = "train",save = False,path = "dataloader_train16_shift",batch_size=self.batch_size)
        dataloader_val = create_dataloader(x=xval,y=yval,mode = "val",save = False,path = "dataloader_val16_shift",batch_size=self.batch_size)
        dataloaders = {"train" : dataloader_train, "val" : dataloader_val}
        
        
        if model_name == "alexnet" :
            model = create_alexnet(n_classes=len(self.dic_emotions),pretrained = True)
            print("Alexnet model with %s classes created"%len(self.dic_emotions))
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        #optimizer = optim.Adam(alexnet.parameters(),lr=0.0001)
        
        if model_path != None :
            loaded = torch.load(model_path)
            loaded.pop(list(loaded.keys())[-1])
            loaded.pop(list(loaded.keys())[-1])
            model.load_state_dict(loaded,strict=False)
            print("Loaded weight from %s"%model_path)
            
            
        print("Starting training \n", '-'*20)
        print("n_epochs={}, batch_size={},lr={},momentum={},val_split={}".format(self.n_epochs,self.batch_size,self.lr,self.momentum,test_index))
        best_model, history = train_model(model,dataloaders,criterion,optimizer,self.n_epochs)
    
        if save_model : 
            torch.save(best_model.state_dict(),save_path)
            np.save(arr = history,file = "%s_history"%save_path)
            print("SAVED as %s and %s_hostiry"%(save_path,save_path))
    
        return best_model, history
    
    
class Losgo_e5(Training):
    
    def __init__(self,n_losgo,data,label, n_epochs, lr, momentum, batch_size, dic_emotions,rcs,noise=None):
        super().__init__(data,label, n_epochs, lr, momentum, batch_size, dic_emotions,rcs,noise)
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
                
            best_model, history = super().start(test_index=test_index,
            model_name = model_name,
            model_path = model_path,
            save_model=save_model,
            save_path="%sbest_model%s"%(save_path,test_index))
                
            best_vals.append(np.max(history[2]))
                
        return best_vals
        
    def eval(self,models_path):
        
        print("EVALUATING MODEL %s AND PLOTTING CONFUSION MATRIX"%models_path)
        mean_cm = np.zeros((len(self.dic_emotions),len(self.dic_emotions)),dtype = np.float64)
    
        for i in range(len(self.kfold)):
            test_index = self.kfold[i]
            #consturciton donnée
            xval = self.data[test_index[0]:test_index[1]]
            yval = self.labels[test_index[0]:test_index[1]]
                
            model_name = "%sbest_model%s"%(models_path,test_index)
            C = eval_model_get_cm(xval,yval,model_name,self.dic_emotions,show_cm= False)
            mean_cm = mean_cm + C
            
        mean_cm = mean_cm/len(self.kfold)
        f, ax = plt.subplots(figsize = (7,7))
        sns.heatmap(mean_cm*100, annot=True, fmt='g', ax=ax,cmap="YlGnBu");
        ax.xaxis.tick_top()
        ax.xaxis.set_ticklabels(self.dic_emotions); 
        ax.yaxis.set_ticklabels(self.dic_emotions);
        
        return mean_cm/len(self.kfold)


class Loso_edb(Training):
    
    def __init__(self,subject_label,data,label, n_epochs, lr, momentum, batch_size, dic_emotions,rcs,noise=None):
        super().__init__(data,label, n_epochs, lr, momentum, batch_size, dic_emotions,rcs,noise)
        self.subject_label = subject_label
        
    def start(self,model_name, model_path, save_model = False,save_path= "" ):
        subject = set(self.subject_label)
        print("subjects: %s"%subject)
        print("CROSS VALIDATION USING LOSO METHOD")
        best_vals = []
        for a in subject : 
            print("SUBJECT LEFT OUT %s"%a)
            test_index = np.where(self.subject_label == a)
                
            best_model, history = super().start(test_index,
            model_name = model_name,
            model_path = model_path,
            save_model=save_model,
            save_path = "%sLOSO_s%s"%(save_path,a))
            
            best_vals.append(np.max(history[2]))
            
        return best_vals
    
    def eval(self,models_path):
        subject = set(self.subject_label)
        print("EVALUATING MODEL %s AND PLOTTING CONFUSION MATRIX"%models_path)
        mean_cm = np.zeros((len(self.dic_emotions),len(self.dic_emotions)),dtype = np.float64)
        for a in subject: 
            test_index = np.where(self.subject_label == a)
            #consturciton donnée
            xval = self.data[test_index]
            yval = self.labels[test_index]
            
            model_name = "%sLOSO_s%s"%(models_path,a)
            C = eval_model_get_cm(xval,yval,model_name,self.dic_emotions,show_cm= False)
            mean_cm = mean_cm + C
        
        mean_cm = mean_cm/len(subject)
        f, ax = plt.subplots(figsize = (7,7))
        sns.heatmap(mean_cm*100, annot=True, fmt='g', ax=ax,cmap="YlGnBu");
        ax.xaxis.tick_top()
        ax.xaxis.set_ticklabels(self.dic_emotions); 
        ax.yaxis.set_ticklabels(self.dic_emotions);
        
        return mean_cm/len(subject)
    
