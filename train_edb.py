# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:20:03 2021

@author: xia-s
"""


import numpy as np

from preprocessing import preprocess_emodb,pcm2float, generate_stft_images
from tests import Loso_edb, plot_cm, Training
from data_augmentation import audio_augmentation

if __name__ == '__main__':
    DTPM_edb = np.array([[84.83,13.1,0.00,0.00,2.07,0.00,0.00],
                         [3.57,80.36,0.00,1.79,12.50,0.00,1.79],
                         [0.00,0.00,88.57,2.86,1.43,4.29,2.86],
                         [0.00,0.00,0.00,93.15,4.11,1.37,1.37],
                         [1.67,8.33,0.00,0.00,88.33,0.00,1.67],
                         [0.00,0.00,0.00,9.09,1.14,86.36,3.41],
                         [2.33,4.65,0.00,0.00,2.33,2.33,88.37]])
    
    dic_emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'boredom','neutral']
    n_classes = len(dic_emotions)
    lr = 0.001
    momentum = 0.9
    n_epochs = 100
    batch_size = 30
    model_name = "alexnet"
    rcs = 13
    noise = None
    sl = 3 #signal length in seconds
    
    Fs,audio_raw,labels,subject_label = preprocess_emodb()
    taille = np.array([len(audio)/Fs for _,audio in enumerate(audio_raw)])
    float_audio = pcm2float(audio_raw)
    t = 0
    for i in range(len(audio_raw)):
        t = t + len(audio_raw[i])/Fs
    if model_name == "inception" :
        size = (300,300)
    elif model_name == "resnet152":
        size = (227,227)
    else :
        size = (227,227)
        
    hop = (sl*Fs)//size[0]
    images = generate_stft_images(float_audio,out_size = size,signal_len=sl,Fs=Fs,n_fft=455,
                                  win_length = 455,hop_length = hop, save = False, toint = True)

    
    # images = np.load("emodb_3s_images.npy",allow_pickle = True)
    # labels = np.load("emodb_labels.npy",allow_pickle = True)
    # subject_label = np.load("emodb_subject_label.npy",allow_pickle = True)
    
    """val split"""
    # test_index = np.where(subject_label == '03')
    # t = Training(dic_emotions, images, labels, n_epochs, lr, momentum, batch_size, rcs = rcs,noise = noise,raw_data =audio_raw)
    # t.start(test_index,model_name,None)


    """loso"""
    # loso = Loso_edb(subject_label, dic_emotions,images,labels, n_epochs, lr, momentum, batch_size,rcs=rcs,noise=noise,raw_data = audio_raw)
    # # loso.set_stft(Fs,sl,1023,hop,1023)
    # # loso.start(model_name = model_name,model_path = None,save_model=True,save_path = "models/LOSO_%s_rcs%s/"%(model_name,rcs))
    # A,B = loso.eval("alexnet",'models/LOSO_alexnet_rcs13/')
    
    # import matplotlib.pyplot as plt
    
    # fig,ax = plt.subplots()
    # ax.hist(labels,bins=13)
    # ax.set_ylabel("Number of utterances")

