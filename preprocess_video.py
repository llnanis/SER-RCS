# -*- coding: utf-8 -*-

import numpy as np

from moviepy.editor import VideoFileClip

import os

def extract_audio_array_from_video(avi_path):
    """
    extract mono audio array from single .avi video. sampling rate = 44100 Hz
    
    Parameters
    ----------
    avi_path : path of the video file (.avi)

    Returns
    -------
    mono : mono array of audio track (nb samples,1)
    default sampling rate
    """
    
    video = VideoFileClip(avi_path)
    audio = video.audio
    audio_array = audio.to_soundarray() #audio array, usually stereo
    mono = np.mean(audio_array,axis=1) #convert to mono
    
    return mono


def preprocess_enterface05(save = False,data_path = "enterface05_data.npy",label_path = "enterface05_label.npy"):
    """
    
    Returns
    -------
    x : size (nb_exemples, length(str))
    contains the path for each .avi file
    
    label : TYPE
    size (nb_exemples)
    contains the emotion label of each exemples

    if save == True:
        returns audio, label
    """
    label = []
    x = []
    name = []
    
    directory = os.getcwd() + "\\database\\eNTERFACE05"

    for subject in os.listdir(directory):

        for emotions in os.listdir(directory + "\\" + subject):
    
            if len( os.listdir(directory + "\\" + subject + "\\" + emotions) ) == 1 :
                 path = directory + "\\" + subject + "\\" + emotions + "\\"
                 video_name = 's' + subject[8:] + '_' + emotions[0:2] + ".avi"
                 avi_path = path + video_name
     
                 x.append(avi_path)
                 label.append(emotions)
                 name.append(video_name)
       
            else :        
                
                for sentence in os.listdir(directory + "\\" + subject + "\\" + emotions):
                
                    path = directory + "\\" + subject + "\\" + emotions + "\\" + sentence + "\\"
                    video_name = 's' + subject[8:] + '_' + emotions[0:2] + '_' + sentence[9:] + ".avi"
                    avi_path = path + video_name

                    x.append(avi_path)
                    #x.append(subject)
                    label.append(emotions)
                    name.append(video_name)
       
    if save:
        print("Reading video file and extracting audio...")
        audio = []
        for i in range(0,len(label),1):
            print(i ," ",x[i])
            audio_path = x[i]
            audio.append(extract_audio_array_from_video(audio_path))
        
        audio = np.array(audio)
        label = np.array(label)
        np.save(data_path,audio)
        np.save(label_path,label)
        print("data saved in %s and label saved in %s"%(data_path,label_path))
        return audio,label
    
    return x,label