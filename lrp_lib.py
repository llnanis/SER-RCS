# -*- coding: utf-8 -*-


"""
inspired from https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/README.md
"""

import numpy
import copy

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision

from fine_tune import generate_single_input

def heatmap(R,sx,sy):

    
    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.xlabel("Time sample")
    plt.ylabel("Frequency bin")
    plt.imshow(R*1.5,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest',origin = "lower")
    plt.show()


def newlayer(layer,g):

    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer


def toconv(layers):

    newlayers = []
    first_linear = True
    
    for i,layer in enumerate(layers):

        if isinstance(layer,nn.Linear):

            newlayer = None

            if first_linear : 
                m,n = 256,layer.weight.shape[0] #512 #256
                newlayer = nn.Conv2d(m,n,6) #7 #6
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,6,6)) #7,7 #6,6
                first_linear = False
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))

            newlayer.bias = nn.Parameter(layer.bias)

            newlayers += [newlayer]

        else:
            newlayers += [layer]

    return newlayers


def lrp_alexnet(alexnet,image,label,dic_emotions,display = False):
    
    print("LRP")
    layers = list(alexnet._modules['features']) + toconv(list(alexnet._modules['classifier']))
    L = len(layers)

    A = generate_single_input(image,mode = 'test')
    B = [A]+[None]*L
    
    for l in range(L): 
        B[l+1] = layers[l].forward(B[l])
    
    scores = numpy.array(B[-1].data.view(-1))
    ind = numpy.argsort(-scores)
    
    print("scores of each classes (prediction)")
    for i in ind[:n_classes]:
        print('%20s (%3d): %6.3f'%(dic_emotions[i],i,scores[i]))
        
        
    T = torch.FloatTensor((1.0*(numpy.arange(n_classes)==0).reshape([1,n_classes,1,1])))
    R = [None]*L + [(B[-1]*T).data]
    
    
    for l in range(0,L)[::-1]:
        
        B[l] = (B[l].data).requires_grad_(True)
        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(3,stride= 2)

        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

            if l < 12:       
                rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
                #rho =  lambda p: p  ; incr = lambda z: z+1e-9
            else : 
                rho = lambda p: p  ; incr = lambda z: z+1e-9

            z = incr(newlayer(layers[l],rho).forward(B[l]))  # step 1
            s = (R[l+1]/z).data                                     # step 2
            (z*s).sum().backward(); c = B[l].grad                  # step 3
            R[l] = (B[l]*c).data                                   # step 4
        
        else:
        
            R[l] = R[l+1]
    
    if display == True:
        
        # plt.figure()
        # plt.imshow(img[:,:,0],cmap=plt.cm.gray,origin = 'lower')
        # plt.xlabel("Time sample")
        # plt.ylabel("Frequency bin")
        #heatmap(numpy.array(R[0][0]).sum(axis=0),3.5,3.5)
        
        

        saliency = numpy.array(R[0][0]).sum(axis=0)
        saliency = saliency/numpy.max(saliency)
        b = 10*((numpy.abs(saliency)**3.0).mean()**(1.0/3))
        from matplotlib.colors import ListedColormap
        my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
        my_cmap[:,0:3] *= 0.85
        my_cmap = ListedColormap(my_cmap)
        extent = (0,img.shape[0],0,img.shape[1])
        
        plt.figure()
        plt.xlabel("Time sample")
        plt.ylabel("Frequency bin")
        plt.title("Ground truth: %s, Predicted label: %s"%(label,dic_emotions[ind[0]]))
        plt.imshow(img[::-1,:,0],cmap=plt.cm.gray,extent = extent,)
        
        plt.colorbar()
        plt.imshow(saliency[::-1,:]*5,cmap = my_cmap,vmin=-b,vmax=b,
                   interpolation='nearest',extent = extent,alpha = 0.3)
        plt.colorbar()
        plt.show()

        
    return R,ind
    

if __name__ == '__main__':

    from fine_tune import create_alexnet
    

    

    """EMO-DB"""
    #dic_emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'boredom','neutral']
    #model_path = "models/LOSO_3s_rcs5/LOSO_s03"
    # images = numpy.load("emodb_3s_images.npy",allow_pickle = True)
    # labels = numpy.load("emodb_labels.npy",allow_pickle = True)#
    # subject_label = numpy.load("emodb_subject_label.npy",allow_pickle = True)
    
    """eNTERFACE05"""
    dic_emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    model_path = "models/best_model_e5"
    images = numpy.load("e5_int_stft_amp2db.npy", allow_pickle = True)
    labels = numpy.load("enterface05_labels.npy",allow_pickle = True)


    n_classes = len(dic_emotions)
    model = create_alexnet(n_classes,pretrained = True)
    model.load_state_dict(torch.load(model_path,map_location = torch.device('cpu')))
    #modelvgg = torchvision.models.vgg16(pretrained=True);
    model.eval()
    
    i = 1263
    
    for j in range(10):
        i += 3
        img = images[i]
        lab = labels[i]
        #print("subject_lab %s"%subject_label[i])
        R,ind = lrp_alexnet(model,img,lab,dic_emotions,display = True)
    
    
    # plt.figure()
    # plt.imshow(img[:,:,1])
    # plt.xlabel("Time Sample")
    # plt.ylabel("Frequency Sample")
    # plt.title("Label: %s, Predicted label: %s"%(lab,dic_emotions[ind[0]]))
    # heatmap(numpy.array(R[0][0]).sum(axis=0),3.5,3.5)