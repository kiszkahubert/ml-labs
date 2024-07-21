import numpy as np
from translate.storage.tmx import tmxfile
from keras.layers import Embedding, LSTM, Input
from keras import Model

def read_tmx_file():
    with open('./files/lab 15/en_pol_corpus.tmx','rb') as fin:
        tmx_file = tmxfile(fin,'en','ar')

    return tmx_file

def replace_with_spaces(string,chars):
    for char in chars:
        string = [s.replace(char,' ') for s in string]
    
    return string

def split_into_words(string):
    string = [line.split(' ') for line in string]
    string = remove_blank_chars(string)
    return string

def remove_blank_chars(string):
    for i, s in enumerate(string):
        string[i] = [word for word in s if len(word) > 0]

    return string

def find_all_unique(strings):
    words = [word for string in strings for word in string]
    words = set(words)
    return words

def make_equal_length(intSentences, needed_length):
    return [sentence + ['\emptyChar']*(needed_length-len(sentence)) for sentence in intSentences]
    
def make_word_lists(data, isTarget = False):
    chars_to_delete = ['!', '?', ':', ',', '.', '(', ')',"'", '‘', '-', '"', '$', ']', '[', '\n', '<', '>','-', '“', '—', ';', '&', '/', '\\', '„', '”']
    data = replace_with_spaces(data,chars_to_delete)
    data = split_into_words(data)
    if isTarget:
        data = [['\start']+d+['\end'] for d in data]
    
    lengths = [len(sentence) for sentence in data]
    max_length = max(lengths)
    data = make_equal_length(data,max_length)
    unique = find_all_unique(data)
    unique = list(unique)
    word_to_int = {word:i for i, word in enumerate(unique)}
    int_to_word = {i:word for i, word in enumerate(unique)}
    return data, word_to_int, int_to_word

def transform_to_ints(data,word_to_int):
    int_data = []
    for sentence in data:
        int_sentence = [word_to_int[word] for word in sentence]
        int_data.append(int_sentence)
    
    return int_data

if __name__ == '__main__':
    tmx_file = read_tmx_file()
    examples = list(tmx_file.unit_iter())
    eng = [ex.source for ex in examples]
    pol = [ex.target for ex in examples]
    
    eng, wordInt_eng, intWord_eng = make_word_lists(eng)
    pol, wordInt_pol, intWord_pol = make_word_lists(pol)
    eng_ints = np.array(transform_to_ints(eng,wordInt_eng))
    pol_ints = np.array(transform_to_ints(pol,wordInt_pol))
    eng_int_in = eng_ints[:,:-1]
    eng_int_out = eng_ints[:,1:]

    output_dim_cnt = 100
    pol_dict_size = len(list(wordInt_pol.values()))
    eng_dict_size = len(list(wordInt_eng.values()))
    max_pl_length = pol_ints.shape[1]
    max_eng_length = eng_int_in.shape[1]
    encoder_lstm = LSTM(output_dim_cnt,return_sequences=False,return_state=True)
    decoder_lstm = LSTM(output_dim_cnt,return_sequences=True,return_state=True)
    input_pl = Input((max_pl_length,))
    output_tensor = Embedding(pol_dict_size,output_dim_cnt)(input_pl)
    _, last_cell_pl, memory_pl = encoder_lstm(output_tensor)
    input_eng = Input(eng_int_in.shape[1])
    english_embedding_layer = Embedding(eng_dict_size,output_dim_cnt)
    embedded_english = english_embedding_layer(input_eng)
    output_eng,_,_ = decoder_lstm(embedded_english,initial_state=[last_cell_pl,memory_pl])
    seq2seq = Model(inputs =[input_pl, input_eng], outputs=output_eng)
    seq2seq.compile(loss='CosineSimilarity', optimizer='RMSprop')
    pol_encoder = Model(inputs=input_pl, outputs=[last_cell_pl,memory_pl])
    input_decoder = Input(1)
    embedded_decoder = english_embedding_layer(input_decoder)
    input_eng_last_cell = Input(output_dim_cnt)
    input_eng_memory = Input(output_dim_cnt)
    output_eng, last_cell_eng,memory_eng = decoder_lstm(embedded_decoder,initial_state=[input_eng_last_cell,input_eng_memory])
    eng_decoder = Model(inputs=[input_decoder,input_eng_last_cell,input_eng_memory],outputs=[output_eng,last_cell_eng,memory_eng])
    

