# -*- coding: utf-8 -*-

import numpy as np

import librosa

from scipy.io import wavfile

import os


def generate_stft_image(audio_array,out_size,signal_len,Fs, n_fft,win_length, hop_length, toint=True):
    #pour enterface05 signal_len = 3, Fs=44100, n_fft = 1024-1, hop_length = 582
    #pour enterface05 signal_len = 3, Fs=16000, n_fft = 1024-1, hop_length = 221
    """
    generate an image given an audio array/vector of audio array
    compute short term fourier transform of audio_array for get frequency approximaly between [0,10kHz] and compute delta, delta delta
    
    Parameters
    ----------
    audio_array : np array 1-D audio array
    
    signal_len : scalar default 3seconds
    time of the final audio -> 
    determine the number of samples of audio_array we used to compute stft, if audio_array is shorter -> padding with 0
    
    Fs : int, default 44100 
    sampling frequency of audio_array
    
    n_fft : int, default 1023
    number of frquency bins used for stft 
    
    hop_length : int, default 582 (228 samples for default nb_samples)
    step for STFT
    
    Returns
    -------
    array of size (227,227,3), containing (stft,delta,delta delta)

    """
    
    nb_samples = int(signal_len * Fs)
    if len(audio_array) < nb_samples:
        audio_array = np.pad(audio_array, (0,nb_samples-len(audio_array)) , 'constant')
    #compute sftf of audio array
    stft = librosa.stft(y=audio_array,n_fft=n_fft, win_length = win_length, hop_length=hop_length)
    print(stft.shape)
    #crop
    filtered_stft = stft[0:out_size[0],0:out_size[1]]   

    #10*log(|stft|)
    #stft_db = librosa.power_to_db(filtered_stft, ref=np.max)
    stft_db = librosa.amplitude_to_db(np.abs(filtered_stft), ref=np.max)
    
    #compute delta and delta delta
    d = librosa.feature.delta(stft_db, order=1)
    dd = librosa.feature.delta(stft_db, order=2)
    
    #stack as image and normalize
    A = np.dstack((stft_db,d,dd))
    #A = np.dstack((stft_db,stft_db,stft_db))
    if toint :
        A = np.round((normalize(A)*255)).astype(np.uint8)
    else :
        A  = normalize(A)

    return A


def generate_stft_images(vector_audio_array,out_size,signal_len,Fs, n_fft,win_length, hop_length, save = False, toint = True):
    """
    generate an array of images from an array of audio 
    call generate_image() for each audio in vector_audio_array

    Parameters
    ----------
    vector_audio_array : array of audio array size (nb_audio_array,)
        DESCRIPTION.
    signal_len : number of seconds of the signal
    number of samples of audio_array we used to compute stft. The default is 3*441000 (3s)
    Fs : TYPE, optional
        Sampling frequency. The default is 44100.
    n_fft : TYPE, optional
        number of bins [0,Fs]. The denpfault is 1024-1 to obtain 512 bins for [0, Fs/2]
    hop_length : TYPE, optional
        step of the analysis. The default is 582.

    Returns
    -------
    array of images (nb_audio, 227,227,3)

    """
    print("Generating STFT Images from audios")
    print("\t out_size %s, signal_len %s, Fs %s, n_fft %s, win_length %s,hop_length %s, save %s, toint %s"%(out_size,signal_len,Fs,n_fft,win_length,hop_length,save,toint))
    res = []
    for i in range(len(vector_audio_array)):
        res.append(generate_stft_image(vector_audio_array[i],out_size,signal_len = signal_len, 
        Fs = Fs, n_fft = n_fft, win_length=win_length,hop_length= hop_length, toint = toint))
    
    
    if save : 
        file_name = "generated_%s_images_stft_%s"%(toint,str(len(res)))
        np.save(arr = np.array(res),file = file_name)
        print("generated images saved as %s.npy"%file_name)
        
    return np.array(res)


def generate_cqt_image(audio_array,out_size,signal_len,Fs,n_bins,bins_per_octave, toint = True):
    #pour enterface signal_len= 3, Fs = 44100, n_bins = 84*3,bins_per_octave = 12*3
    
    nb_samples = int(signal_len * Fs)
    
    if len(audio_array) < nb_samples:
        audio_array = np.pad(audio_array, (0,nb_samples-len(audio_array)) , 'constant')
        
    #compute sftf of audio array
    cqt = librosa.cqt(y=audio_array,n_bins=n_bins, bins_per_octave = bins_per_octave)

    #crop
    filtered_cqt = cqt[0:out_size[0],0:out_size[1]]   

    #10*log(|stft|)
    #cqt_db = librosa.power_to_db(filtered_cqt, ref=np.max)
    cqt_db = librosa.amplitude_to_db(np.abs(filtered_cqt), ref=np.max)
    
    #compute delta and delta delta
    d = librosa.feature.delta(cqt_db, order=1)
    dd = librosa.feature.delta(cqt_db, order=2)
    
    #stack as image and normalize
    A = np.dstack((cqt_db,d,dd))
    #A = np.dstack((stft_db,stft_db,stft_db))
    if toint :
        A = np.round((normalize(A)*255)).astype(np.uint8)
    else :
        A  = normalize(A)

    return A


def generate_cqt_images(vector_audio_array,out_size,signal_len,Fs,n_bins,bins_per_octave, save = False,toint = True):
    print("Generating CQT Images from audios")
    print("\t signal_len %s, Fs %s, n_bins %s, bins_per_octave %s, save %s, toint %s"%(signal_len,Fs,n_bins,bins_per_octave,save,toint))
    res = []
    for i in range(vector_audio_array.shape[-1]):
        res.append(generate_cqt_image(vector_audio_array[i], out_size,signal_len = signal_len, Fs = Fs, n_bins = n_bins, bins_per_octave =bins_per_octave, toint = toint))
        print(i)
    if save : 
        file_name = "generated_%s_images_cqt_%s"%(toint,str(len(res)))
        np.save(arr = np.array(res),file = file_name)
        print("generated images saved as", file_name, ".npy")
        
    return np.array(res)    
    
    
def normalize(A):
    """
    A array of size (227,227,3)
    normalize one image A between [0,1]
    normalize along each depth independetly
    
    """
    normalized = np.empty(shape = A.shape, dtype = np.double)
    for i in range(A.shape[-1]):
        normalized[:,:,i] = (A[:,:,i]-np.min(A[:,:,i]))  / (np.max(A[:,:,i])-np.min(A[:,:,i]))
        
    return normalized
   
    
def pcm2float(audio_array):
    """
    convert fixed point value of audio_array into float point.
    """
    try : res = audio_array.copy()
    except : res = audio_array 
        
    for i in range(len(audio_array)):
        res[i] = res[i]/ (2**15)
    return res


def mel_map(audio_array,Fs):
    
    """
    
    compute static mel-spectrogram, delta and delta delta given an audio array.

    Parameters
    ----------
    audio_array : 1-D array

    Returns
    -------
    static, delta, delta delta mel-spectrogram of audio array.

    """
    
    #compute static mel with 64 banks freq, [20,8000]Hz, 25ms hamming window size, 10 ms overlapping.
    mel = librosa.feature.melspectrogram(y=audio_array,sr=Fs,n_mels=64,fmin=20,fmax=8000,hop_length=(int)((10/1000)*Fs),n_fft=(int)((25/1000)*Fs))
    
    #to db
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    #compute delta and delta delta
    d = librosa.feature.delta(mel_db, order=1)
    dd = librosa.feature.delta(mel_db, order=2)
    
    
    
    return np.array([mel_db,d,dd])


def overlap(A, context_win_size = 64, frame_shift = 30 ):

    """
    Compute overlapping segments of A

    Parameters
    ----------
    mel : matrix size (number mels, number of samples)

    context_win_size : int, number of frames considered in the window
    
    frame_shift : step to take for next window 
        
    Returns
    -------
    overlapped : size (number segments, A[0], context_win_size)
    """
    overlapped = []
    padding = np.zeros( (A.shape[0],context_win_size-(A.shape[1]%frame_shift)))
    B = np.hstack((A,padding))
    for i in range(A.shape[1]%frame_shift):
        temp = B[0:B.shape[0],i*30:i*30+context_win_size]
        overlapped.append(temp)
    overlapped = np.array(overlapped)
    
    return overlapped


def preprocess_emodb(save = False, data_path = "emodb_audio.npy", label_path = "emodb_labels.npy"):
    
    """
    preprocess the wav files in emo_db database in path database/emo_db_berlin/wav
    Return :
        sr : sampling rate, int
        data : raw audio extracted from database, data is an array of object since each audio array are not of the same length
        label : array of strings containing the labels of each audio array
        subject_label : array of strings indicating the subject 
    """
    dic = dict()
    dic['W'],dic['L'],dic['E'],dic['A'],dic['F'],dic['T'],dic['N'] = 'anger','boredom','disgust','fear','happiness','sadness','neutral' 
    label = []
    data = []
    subject_label = []
    audio_files = []

    try : 
        
        directory = "%s\\database\\emo_db_berlin\\wav"%os.getcwd()
        for audio_file in os.listdir(directory):
            sr , audio = wavfile.read("%s\\%s"%(directory,audio_file))
            data.append(audio)
            audio_files.append(audio_file)
            subject_label.append(audio_file[0:2])
            label.append(dic[audio_file[5]])
        print("WINDOWS PATH")
        
    except :
        
        directory = "%s/database/emo_db_berlin/wav"%os.getcwd()
        for audio_file in os.listdir(directory):
            sr , audio = wavfile.read("%s/%s"%(directory,audio_file))
            data.append(audio)
            audio_files.append(audio_file)
            subject_label.append(audio_file[0:2])
            label.append(dic[audio_file[5]])
        print("LINUX PATH")
        
    data = np.array(data,dtype = object)
    label = np.array(label)
    subject_label = np.array(subject_label)
    print("SAMPLING RATE = %s Hz"%sr)
    
    # print("10 premiers :")
    # for i in range(10):
    #     print(audio_files[i],label[i],subject_label[i])
    # print("10 derniers : ")
    # for i in range(len(data)-10,len(data),1):
    #     print(audio_files[-i],label[-i],subject_label[-i])
    
    if save : 
        print("data saved in %s and label saved in %s"%(data_path,label_path))
        np.save(file = data_path,arr = data)
        np.save(file = label_path,arr = label)
        np.save(file = "emo_db_subjects",arr = subject_label)

    return sr,data,label,subject_label
