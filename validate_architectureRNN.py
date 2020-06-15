import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import gc
import re
from scipy.io import wavfile
from scipy import signal


train_data_path = "data/train/train/audio/"
test_data_path = "data/test/test/audio/"

print(os.listdir("./"))

train_labels = 'yes no up down left right on off stop go silence unknown'.split()

def log_specgram(audio, sample_rate, window_size=20,

                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in train_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))

labels, fnames = list_wavs_fname(train_data_path)

L = 16000
new_sample_rate = 8000
Y_train = []
X_train = []

for label, fname in zip(labels, fnames):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    if len(samples) > 16000:
        n_samples = chop_audio(samples)
    else: n_samples = [samples]
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        Y_train.append(label)
        X_train.append(specgram)
X_train = np.array(X_train)
X_train = X_train.reshape(tuple(list(X_train.shape) + [1]))
Y_train = label_transform(Y_train)
label_index = Y_train.columns.values
Y_train = Y_train.values
Y_train = np.array(Y_train)

del labels, fnames
gc.collect()

print(X_train.shape)

def test_data_generator(batch=16):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        imgs.append(specgram)
        fnames.append(path.split('\\')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
        yield fnames, imgs
    return

from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, BatchNormalization, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import tensorflow as tf   
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import tensorflow.keras.layers as layers

def reset_weights(model):
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run()
            
def validate_architectureRNN(architecture_name,
                          num_epochs,
                          num_iterations):
    
    for it in range(num_iterations):
    
        # Reinitializing  weights
        # reinitializing weights not working on CPU
        
        input_shape = (99, 81)
        nclass = len(train_labels)
        inp = layers.Input(shape=input_shape)
        norm_inp = layers.BatchNormalization()(inp)
        lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(norm_inp)
        lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(norm_inp)
        lstm = layers.GlobalMaxPooling1D()(lstm)
        dense_1 = layers.Dense(128, activation=activations.relu)(lstm)
        dense_out = layers.Dense(nclass, activation=activations.softmax)(dense_1)

        model = models.Model(inputs=inp, outputs=dense_out)
        opt = optimizers.Adam()

        model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['accuracy'])
    
        x_train, x_valid, y_train, y_valid = train_test_split(X_train,
                                                              Y_train,
                                                              test_size=0.1)
        
        mc = ModelCheckpoint(f'./results/best_model_{architecture_name}_{it}.h5',
                     monitor='val_accuracy',
                     mode='max',
                     verbose=1,
                     save_weights_only=True,
                     save_best_only=True)
        
        history = model.fit(x_train,
                            y_train,
                            batch_size=32,
                            validation_data=(x_valid, y_valid),
                            epochs=num_epochs,
                            shuffle=True,
                            callbacks=[mc])
        
        history = history.history
        
        # saving json
        with open(f'./results/results_{architecture_name}_{it}', 'w') as fp:
            json.dump(history, fp, indent=4)
            
            
        # prediction
        index = []
        results = []
        
        for fnames, imgs in tqdm(test_data_generator(batch=32)):
            predicts = model.predict(imgs)
            predicts = np.argmax(predicts, axis=1)
            predicts = [label_index[p] for p in predicts]
            index.extend([text.split("/")[-1] for text in fnames])
            results.extend(predicts)

            df = pd.DataFrame(columns=['fname', 'label'])
            df['fname'] = index
            df['label'] = results
            df.to_csv(f'./results/submission_{architecture_name}_{it}.csv', index=False)

validate_architectureRNN("BidirectionalLSTM", 1, 1)

