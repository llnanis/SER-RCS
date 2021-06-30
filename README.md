# stage_ser
 sylvain_rep
 
 Scripts :
 + preprocessing_video.py : used to extract audio from eNTERFACE05 data-set.
 + preprocessing.py : used to preprocess the audio arrays. Compute STFT and generate 3-D tensor as np.darray().
 + data_augmentation.py : used to augment our data, contains noise adding and RCS.
 + fine_tune.py : used to create DCNNs model from pytroch, create DataLoader(), link between numpy and torchvision framework.
 + tests.py : implementation of different training setups such as simple data split, LOSGO (Leave One Speaker Group Out) and LOSO (Leave One Speaker Out).
 + train_e5.py : script to load datas, instantiate a model and train on eNTERFACE05 data-set (either training on a simple split or by using LOSGO).
 + train_edb.py : cript to load datas, instantiate a model and train on EMO-DB data-set (either training on a simple split or by using LOSGO).

Dependecies :
 + Python (3.8.5)
 + moviepy (1.0.3) used in preprocessing_vide.py
 + librosa (0.8.0)
 + Pytorch (1.8.1) and torchvision (0.9.1)
 + CUDA to train on GPU
