import pandas as pd
import os
import collections
import spacy
import pickle
from spacy.tokenizer import Tokenizer
import numpy as np
import pickle
import re
import math
import concurrent.futures

nlp = spacy.load('en')

# load translators
name = "content"
PKL_SAVE = f"{name}.pkl"
with open(os.path.join("pickles",PKL_SAVE), "rb") as pkl_file:
        contents, content_vocab, content_word_dict = pickle.load(pkl_file)
PUBLICATION = "NPR"        
DATA = contents
CONTENT_WORD_DICT = content_word_dict
N_VOCAB = len(content_vocab)

def process_i(index):
    step_token = 1
    global CONTENT_WORD_DICT, N_VOCAB, TRAINING_DATA, STEP_DATA, SEQ_LENGTH, PUBLICATION
    temp_ct = int(index / STEP_DATA)
    i = index
    print(f"iteration {temp_ct}")

    contents = TRAINING_DATA[i:i+STEP_DATA]
    sequences = []
    next_words = []
    content_words = [word.text for doc in nlp.pipe(contents, batch_size=STEP_DATA) for word in doc if word.is_alpha or word.is_punct]
    len_content_words = len(content_words)
    for j in range(0, len_content_words - SEQ_LENGTH, step_token):
        sequence = content_words[j:j+SEQ_LENGTH]
        next_word = content_words[j+SEQ_LENGTH]
        sequences.append(sequence)
        next_words.append(next_word)
    training = encode_sequences(sequences, CONTENT_WORD_DICT, SEQ_LENGTH, N_VOCAB)
    targets =  encode_next_words(next_words, CONTENT_WORD_DICT, N_VOCAB)
    training_data = (training, targets)
    filepath = os.path.join(os.getcwd(), f"data/training/training_{PUBLICATION}_{temp_ct}.npy")
    with open(filepath, "wb") as file:
        np.save(file, training, allow_pickle=True)
    print(f"SAVED training{temp_ct}.npy")
    filepath = os.path.join(os.getcwd(), f"data/target/target_{PUBLICATION}_{temp_ct}.npy")
    with open(filepath, "wb") as file:
        np.save(file, targets, allow_pickle=True)

    print(f"SAVED targets{temp_ct}.npy")
    del next_words, sequences, content_words, training, targets
    return i



def encode_sequences(sequences, word_dict, seq_length, n_vocab):
        size = 8 * len(sequences) * seq_length * n_vocab / 1000000
        print(f"{size} MB storage required")
        data = np.zeros(shape=(len(sequences), seq_length, n_vocab), dtype=np.int8)
        for sequence in sequences:
            if len(sequence) > seq_length:
                sequence = sequence[:seq_length]
            elif len(sequence) < seq_length:
                raise NotImplementedError(f"Need a sequence of length {seq_length}")
        for i, sequence in enumerate(sequences):
            for j,word in enumerate(sequence):
                data[i, j, word_dict[word.lower()]] = 1

        return(data)

def encode_next_words(next_words, word_dict, n_vocab):
    next_word_encode = np.zeros(shape=(len(next_words), n_vocab), dtype=np.int8)
    for i,next_word in enumerate(next_words):
        next_word_encode[i, word_dict[next_word]] = 1
    return(next_word_encode)

def fit_gen2(word_dict, seq_length, n_vocab, step_data=100, step_token=1):
        print(f"fit gen called for {seq_length} length sequences")
        global DATA, TRAINING_DATA, VALIDATION_DATA, VALIDATION_INDEX, COUNTER

        np.random.shuffle(DATA)
        TRAINING_DATA = DATA
        num_seq = len(TRAINING_DATA)
        total_iterations = math.floor((num_seq - step_data) // step_data)
        
        print("here")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                result = executor.map(process_i, range(0, num_seq - step_data, step_data))
                for i in result:
                        print(i)

        return True
                                                                                                                                                                                                                                                                    
SEQ_LENGTH = 15
batch_size = 16
STEP_DATA = batch_size

if __name__ == "__main__":
    fit_gen2(content_word_dict, SEQ_LENGTH, n_vocab=len(content_word_dict), step_data=batch_size)
