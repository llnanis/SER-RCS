    # -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import torch

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddGaussianSNR


def random_shift_np(image,n_shift,display = False):
    """
    random shift image n_shift times.

    Parameters
    ----------
    image : RGB (227,227,3)
        DESCRIPTION.
    n_shift : int
        number of shift wanted
    display : bool 
        display shifted images

    Returns
    -------
    shifted : array of image (n_shift,227,227,3)
    shift : array of int (n_shift) contains value of the shift
        
    """
    shift = np.random.randint(low=1,high = image.shape[0]-1,size = n_shift)
    shifted = np.empty(shape = (n_shift,image.shape[0],image.shape[1],3), dtype = image.dtype)
    
    for i in range(n_shift):
        shifted[i,:,shift[i]:,:] = image[:,0:image.shape[0]-shift[i],:]
        shifted[i,:,0:shift[i],:] = image[:,image.shape[0]-shift[i]:,:]
        
        if display :
            plt.figure()
            plt.imshow(shifted[i])
            plt.title("shift of %s"%shift[i])
            plt.xlabel("Time sample")
            plt.ylabel("Frequency bin")
        
    return shifted, shift


def random_shift_torch(image,n_shift,display = False):
    """
    torch version of random_shift_np
    Returns
    -------
    shifted : tensor of image (n_shift,3,227,227)


    """
    shift = np.random.randint(low=1,high = 226,size = n_shift)
    shifted = torch.empty(size = (n_shift,3,227,227))
    
    for i in range(n_shift):
        shifted[i,:,:,shift[i]:] = image[:,:,0:227-shift[i]]
        shifted[i,:,:,0:shift[i]] = image[:,:,227-shift[i]:]
        
        if display : 
            plt.figure()
            plt.imshow( shifted[i].permute(1, 2, 0)  )
            plt.title(str(i)+" "+str(shift[i]))
            
    return shifted


def random_shift_data(xtrain,ytrain, n_shift,save = False):
    """
    random shift the train data xtrain and create corresponding labels.

    Parameters
    ----------
    xtrain : train set array of images (n_exemples,227,227,3)
    ytrain : array containing the label of train set (n_exemples)
    n_shift : int, number of shift
    save : BOOL, save the generated shifted data 

    Returns
    -------
    data : array of image containing original xtrain and shifted images (n_exemples + (n_shift*n_exemples), 227,227,3)
    labels : array int containing the corresponding labels of data
    
    THE SHIFTED DATA ARE APPENED AT THE END OF XTRAIN
    """
        
    print("Random Shifting, {} shift per image".format(n_shift))
    if (n_shift <= 0) :
        print("Number of Shift < 0, NO RCS APPLIED")
        return xtrain,ytrain
    ###################################################################################################
    if (n_shift < 1) :
        proba = np.random.uniform(low=0,high=1,size=len(ytrain))
        indexs = np.where(proba<=n_shift)
        
        x_augm = []
        y_augm = []
        for i in range(len(indexs[0])):
            j = indexs[0][i]
            shift,_ = random_shift_np(image = xtrain[j],n_shift = 1 )
            x_augm.append(shift[0])
            y_augm.append(ytrain[j])
        
        x_augm = np.array(x_augm,dtype=xtrain.dtype)
        y_augm = np.array(y_augm)
    
            
    ##################################################################################################    
    else : 
        x_augm = np.empty(shape = (len(ytrain)*n_shift,xtrain.shape[1],xtrain.shape[2],3),dtype=xtrain.dtype)
        y_augm = np.empty(shape = (len(ytrain)*n_shift),dtype=ytrain.dtype)
    
        for i in range(len(ytrain)):
             shift , _ = random_shift_np(image = xtrain[i],n_shift = n_shift)    
             x_augm[i*n_shift:i*n_shift+n_shift,:,:,:] = shift
             y_augm[i*n_shift:i*n_shift+n_shift] = ytrain[i]
            
    ######################################################################################################
    
    data = np.vstack((xtrain,x_augm))
    labels = np.append(ytrain,y_augm)
        
    if save : 
            
        dataname = "xshift_%s" %n_shift
        labelname = "yshift_%s" %n_shift
            
        np.save(arr=data,file = dataname)
        np.save(arr=labels,file = labelname)
        print("Random shift saved as {}.npy and {}.npy".format(dataname,labelname))
        
    return data,labels


def add_gaussian_noise_to_images(xtrain,mean,var,normalized = False,save = False):
    """
    add gaussian noise to xtrain

    Parameters
    ----------
    xtrain : train set array of images (n_exemples,227,227,3)
    mean : mean of gaussian 
    var : var of gaussian
    normalized : BOOL
        whether xtrain is normalized (float[0 1]) or RGB(int [0 255])
        False = RGB, True = NORMALIZED
    save : BOOL, save the generated data
    
    Returns
    -------
    xnoise : data with gaussian noise added

    """
    print("Adding Gaussian Noise")
    if normalized :
        sigma = var**0.5
    else : 
        sigma = (var*255)**0.5
        
    noise = np.random.normal(loc = mean, scale = sigma, size = xtrain.shape)
    xnoise = xtrain + np.round(noise).astype(xtrain.dtype)
    xnoise[np.where(xnoise<0)] = 0
    xnoise[np.where(xnoise>255)] = 255
    
    if save : 
        
        if len(xtrain <= 1053) :
            filename = "xnoise_%s_%s"%(mean,var)
        else :
            filename = "xshiftednois_%s_%s"%(mean,var)
            
        np.save(arr=xnoise,file = filename)
        print("Gaussian noise data saved as {}.npy".format(filename))
        
    return xnoise


def puissance(audio):
    """compute the power of an audio signal
    audio : array 
    return scalar
    """
    return np.mean(np.square(abs(audio)))


def snr_db(signal,noisy_signal):
    """compute SNR_db ratio given 2 arrays
    signal : clean signal
    noisy_signal : signal with noise
    
    both signal can be array of audioS
    """
    try : 
        snr = puissance(signal)/puissance(signal-noisy_signal)
        return 10*np.log10(snr)
    except :
        snrs = []
        for i in range(len(signal)):
            snrs.append(puissance(signal[i])/puissance(signal[i]-noisy_signal[i]))
        return snrs

def linear(snr_db) :
    return 10**(snr_db/10)

def db(snr):
    return 10*np.log10(snr)

def add_noise(SNR_db,audio):
    """
    add noise to audio, noise is added given SNR_db
    Parameters
    ----------
    SNR_db : scalar, wanted SNR (db) ratio 
    audio : array, audio signal considered

    Returns
    -------
    array, audio signal with noise

    """
    ex1 = puissance(audio)
    snr = 10**(SNR_db/10)
    ex2 = ex1/snr
    noise = np.random.normal(loc = 0, scale = ex2**(1/2), size = audio.shape)
    #print("noise power wanted: ",10*np.log10(ex2))
    #print("noise power obtained: ",10*np.log10(puissance(noise)))
    try : 
        return audio.copy()+noise
    except :
        return audio + noise
    
    
def add_gaussian_noise_to_audios(audios,train_size,SNR_db,p=0.5):
    """
    add noise to an array of audios
    call function add_noise()
    
    Parameters
    ----------
    audios : array, (nb_audio,)
    SNR_db : scalar, SNR (db) ratio wanted
    p : fscalar, probability to add noise to each audio in audios

    Returns
    -------
    noisy : array, (nb_audio,) contains new audios, some with noise
    """
    
    nb_samples = 3* 44100
    print("Adding gaussian noise to audios with SNR of %sdB with probability of %s, nb_samples %s"%(SNR_db,p,nb_samples) )
    
    proba = np.random.uniform(low=0,high=1,size=train_size)
    try : 
        noisy = audios.copy() ##BECAUSE AUDIOS CAN BE AN ARRAY OF OBJECT!!!!!!
    except:
        noisy = audios
        
    indexs = np.where(proba<=p)
    for i in range(len(indexs[0])):
        

        j = indexs[0][i]
        if len(noisy[j]) < nb_samples:
            noisy[j] = np.pad(noisy[j], (0,nb_samples-len(noisy[j])) , 'constant')
            
        noisy[j] = add_noise(SNR_db,noisy[j])

    return noisy

def audio_augmentation(xtrain,ytrain,Fs,p=0.5):
    
    proba = np.random.uniform(low=0,high=1,size=len(xtrain))

    augment = Compose([
        AddGaussianSNR(min_SNR=1/10, max_SNR=1/10, p=1),#20db
        #TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
        #PitchShift(min_semitones=-4, max_semitones=4, p=1),
        #Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
    ])
    indexs = np.where(proba<=p)
    xa = []
    ya = []
    print("DATA AUGMENTATION ON AUDIO, NUMBER OF AUGMENTED DATA %s"%len(indexs[0]))
    
    for i in range(len(indexs[0])):
        
        j = indexs[0][i]
        
        try : 
            xa.append(  augment( samples = xtrain[j].copy(), sample_rate = Fs) )
        except :
            xa.append(  augment( samples = xtrain[j], sample_rate = Fs) )
        
        ya.append( ytrain[j])

    # x = np.append(xtrain,np.array(xa,dtype = object),axis=0)
    # y = np.append(ytrain,np.array(ya))
    
    return np.array(xa,dtype = object) , np.array(ya)
        