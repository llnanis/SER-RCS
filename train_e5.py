# -*- coding: utf-8 -*-

import numpy as np

from tests import Losgo_e5, Training, plot_cm
from preprocessing import generate_stft_images
from data_augmentation import audio_augmentation

if __name__ == '__main__':
    
    DTPM_e5 = np.array([[87.50,2.50,5.00,1.00,1.00,3.00],
                        [2.08,78.13,4.17,9.38,3.65,2.60],
                        [2.97,6.93,73.27,4.46,4.95,7.43],
                        [2.02,7.07,1.52,81.31,2.02,6.06],
                        [1.72,5.17,10.34,2.16,75.00,5.60],
                        [3.98,2.84,3.98,2.84,1.70,84.66]])
    
    
    #plt.imshow( tensor_image.permute(1, 2, 0)  )
    
    dic_emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    n_classes = len(dic_emotions)
    lr = 0.001
    momentum = 0.9
    n_epochs = 60
    batch_size = 32
    model_name = "alexnet"
    rcs = 41
    noise = None
    sl = 3 #signal length in seconds
    
    #LOAD DATA AND LABELS
    audio_raw = np.load("enterface05_audio.npy",allow_pickle = True)
    Fs = 44100
    
    if model_name == "inception" :
        size = (300,300)
    elif model_name == "resnet152":
        size = (227,227)
    else :
        size = (227,227)
    
    hop = (sl*Fs)//size[0]
    images = generate_stft_images(audio_raw,out_size = size,signal_len = 3,Fs=Fs,
    n_fft = 1023, hop_length = hop, win_length = 1023, save = False, toint = True)
    
    # images = generate_stft_images(audio_raw,out_size = (300,300),signal_len = 3,Fs=44100,
    # n_fft = 1023, hop_length = 441, win_length = 1023, save = True, toint = True)
    
    # images = generate_cqt_images(audio_raw,out_size = input_size,signal_len = 3, Fs = 44100,
    # n_bins = 84*6 ,bins_per_octave = 12*6 ,save = True, toint = True)
    
    #images = np.load("e5_int_stft_amp2db.npy")
    #images = np.load("e5_int_stft_pow2db.npy")
    #images = np.load("e5_int_cqt_amp2db.npy")
    #images = np.load("e5_cqt_84x6.npy")
    #images = np.load("e5_int_stft_inception.npy")
    
    labels = np.load('enterface05_labels.npy',allow_pickle = True)
    
    
    """split test"""
    # test_index =[1053, 1293]  #1053 -> ,1-36 on train and 37-44 on val 18%
    # t = Training(dic_emotions,images, labels, n_epochs, lr, momentum, batch_size,rcs = rcs,noise=noise,raw_data=audio_raw)
    # t.set_stft(Fs=Fs,signal_len=sl,n_fft = 1023,hop_length = hop,win_length = 1023)
    # m,h,duration = t.start(test_index,model_name,None,True,"models/e5_%s_rcs%s"%(model_name,rcs))
    
    # #t.eval(test_index,"models/best_model_e5")
    
    
    """losgo"""
    losgo = Losgo_e5(5,dic_emotions,images, labels, n_epochs, lr, momentum, batch_size, rcs = rcs,noise = None)
    # losgo.start(model_name=model_name,model_path = None,save_model = True,save_path = "models/LOSGO_%s_rcs%s/"%(model_name,rcs))
    a,b = losgo.eval("alexnet","models/LOSGO_alexnet_rcs41_v0/")
    
    
    
    # import matplotlib.pyplot as plt
    
    # fig,ax = plt.subplots()
    # ax.hist(labels,bins=11)
    # ax.set_ylabel("Number of utterances")
