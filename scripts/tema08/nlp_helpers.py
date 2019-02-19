#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:04:22 2019
    Biblioteca de NLP del curso de TensorFlow
@author: juangabriel
"""

import os
import urllib.request
import io
import tarfile
import string
import collections
import numpy as np




def load_movies_data():
    save_folder_name = "../../datasets/movies_data"
    pos_file = os.path.join(save_folder_name, 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polarity.neg')
    
    if os.path.exists(save_folder_name):
        ## Podemos cargar la info directamente desde el PC
        pos_data = []
        with open(pos_file, 'r') as temp_pos_file:
            for row in temp_pos_file:
                pos_data.append(row)
                
        neg_data = []
        with open(neg_file, 'r') as temp_neg_file:
            for row in temp_neg_file:
                neg_data.append(row)
        
    else:
        ## Debemos descargar los ficheros de internet y guardarlos en esta carpeta
        url = "http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
        stream_data = urllib.request.urlopen(url)
        tmp = io.BytesIO()
        while True:
            s = stream_data.read(16384)
            if not s: 
                break
            tmp.write(s)
        stream_data.close()
        tmp.seek(0)
        
        tar_file = tarfile.open(fileobj=tmp, mode='r:gz')
        pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
        neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
        
        pos_data = []
        for line in pos:
            pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
            
        neg_data = []
        for line in neg:
            neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
            
        tar_file.close()
        
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)
        with open(pos_file, 'w') as pos_file_handler:
            pos_file_handler.write(''.join(pos_data))
        with open(neg_file, 'w') as neg_file_handler:
            neg_file_handler.write(''.join(neg_data))
    
    texts = pos_data + neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    return (texts, target)  




def normalize_text(texts, stops):
    texts = [x.lower() for x in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    texts = [' '.join(word for word in x.split() if word not in (stops)) for x in texts]
    texts = [' '.join(x.split()) for x in texts]
    return texts




def build_dictionary(sentences, vocabulary_size):
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    count = [['RARE', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return word_dict




def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        for word in sentence:
            if word in word_dict:
                word_ix = word_dict[word]#posición/ID de la palabra en el word dict
            else:
                word_ix = 0 ##posición/ID de la palabra RARE
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return data





def generate_batch_data(sentences, batch_size, window_size, method = 'skip_gram'):
    '''
        Skip Gram: Mi perro come su comida -> (Mi, come), (perro, come), (su, come), (comida, come)
        Cbow: Mi perro come su comida -> ([Mi,perro,su,comida]; come)
    '''
    batch_data = []
    label_data = []
    
    while len(batch_data) < batch_size:
        # seleccionamos una frase aleatoria del conjunto de sentences
        rand_sentence_ix = int(np.random.choice(len(sentences), size = 1))
        rand_sentences = sentences[rand_sentence_ix]
        
        window_seq = [rand_sentences[max((ix-window_size),0):(ix+window_size+1)] 
                      for ix, x in enumerate(rand_sentences)]
        label_idx = [ix if ix < window_size else window_size for ix, x in enumerate(window_seq)]
        
        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y]+x[(y+1):]) for x,y in zip(window_seq, label_idx)]
            # Convertir el dato a una lista de tuplas (palabra objetivo, contexto)
            tuple_data = [(x,y_) for x, y in batch_and_labels for y_ in y]
            batch, labels = [list(x) for x in zip(*tuple_data)]

        elif method=='cbow':
            batch_and_labels = [(x[:y]+x[(y+1):], x[y]) for x,y in zip(window_seq, label_idx)]
            # Conservar las ventanas de tamaño 2*window_size
            batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x)==2*window_size]
            
            if len(batch_and_labels) <1:
                continue
            
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method == 'doc2vec': 
            #Elegimos las windowsize palabras anteriores a una dada
            batch_and_labels = [(rand_sentences[i:i+window_size], rand_sentences[i+window_size]) for i in range(0, len(rand_sentences)-window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError("Método {} no implementado".format(method))
        
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
        
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return (batch_data, label_data)