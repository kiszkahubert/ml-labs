import numpy as np
from translate.storage.tmx import tmxfile
from keras.layers import Embedding, LSTM, Input
from keras import Model
from tqdm.notebook import tqdm

def read_tmx_file():
    with open('./files/lab 15/en_pol_corpus.tmx','rb') as fin:
        tmx_file = tmxfile(fin, 'en', 'ar')

    return tmx_file

def replace_with_spaces(string, chars):
    for char in chars:
        string = [s.replace(char, ' ') for s in string]

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
    return [sentence + ['\emptyChar'] * (needed_length - len(sentence)) for sentence in intSentences]

def make_word_lists(data, isTarget=False):
    chars_to_delete = ['!', '?', ':', ',', '.', '(', ')', "'", '‘', '-', '"', '$', ']', '[', '\n', '<', '>', '-', '“', '—', ';', '&', '/', '\\', '„', '”']
    data = replace_with_spaces(data, chars_to_delete)
    data = split_into_words(data)
    if isTarget:
        data = [['\start'] + d + ['\end'] for d in data]

    lengths = [len(sentence) for sentence in data]
    max_length = max(lengths)
    data = make_equal_length(data, max_length)
    unique = find_all_unique(data)
    unique = list(unique)
    word_to_int = {word: i for i, word in enumerate(unique)}
    int_to_word = {i: word for i, word in enumerate(unique)}
    return data, word_to_int, int_to_word

def transform_to_ints(data, word_to_int):
    int_data = []
    for sentence in data:
        int_sentence = [word_to_int[word] for word in sentence]
        int_data.append(int_sentence)
        
    return int_data

def make_batch(eng, pol_in, pol_out, weights, batch_size):
    inds = np.arange(eng.shape[0])
    inds = np.random.choice(inds, batch_size, replace=False)
    return eng[inds, ...], pol_in[inds, ...], pol_out[inds, ...], weights[inds, ...]

def vect2word(embedded_word, embedded_dict, int_word_dict):
    embedded_word = embedded_word.squeeze()
    dot_prods = np.dot(embedded_dict, embedded_word)
    dot_prods /= np.linalg.norm(embedded_word)
    dot_prods /= np.linalg.norm(embedded_dict, axis=1)
    closest_int = int(dot_prods.argmax())
    return int_word_dict[closest_int]

def embed(Embedder, dict_int):
    keys = np.array(list(dict_int.keys()))
    embedded = Embedder.predict(keys)
    return embedded

if __name__ == '__main__':
    tmx_file = read_tmx_file()
    examples = list(tmx_file.unit_iter())
    eng = [ex.source for ex in examples]
    pol = [ex.target for ex in examples]
    
    pol, wordInt_pol, intWord_pol = make_word_lists(pol)
    eng, wordInt_eng, intWord_eng = make_word_lists(eng, isTarget=True)
    pol_ints = np.array(transform_to_ints(pol, wordInt_pol))
    eng_ints = np.array(transform_to_ints(eng, wordInt_eng))
    pol_int_in = pol_ints[:, :-1]
    pol_int_out = pol_ints[:, 1:]

    output_dim_cnt = 100
    pol_dict_size = len(list(wordInt_pol.values()))
    eng_dict_size = len(list(wordInt_eng.values()))
    max_pl_length = pol_int_in.shape[1]
    max_eng_length = eng_ints.shape[1]
    encoder_lstm = LSTM(output_dim_cnt, return_sequences=False, return_state=True)
    decoder_lstm = LSTM(output_dim_cnt, return_sequences=True, return_state=True)
    input_eng = Input((max_eng_length,))
    output_tensor = Embedding(eng_dict_size, output_dim_cnt)(input_eng)
    _, last_cell_eng, memory_eng = encoder_lstm(output_tensor)
    input_pol = Input(pol_int_in.shape[1])
    polish_embedding_layer = Embedding(pol_dict_size, output_dim_cnt)
    embedded_polish = polish_embedding_layer(input_pol)
    output_pol, _, _ = decoder_lstm(embedded_polish, initial_state=[last_cell_eng, memory_eng])
    seq2seq = Model(inputs=[input_eng, input_pol], outputs=output_pol)
    seq2seq.compile(loss='CosineSimilarity', optimizer='RMSprop')
    eng_encoder = Model(inputs=input_eng, outputs=[last_cell_eng, memory_eng])
    input_decoder = Input(1)
    embedded_decoder = polish_embedding_layer(input_decoder)
    input_pol_last_cell = Input(output_dim_cnt)
    input_pol_memory = Input(output_dim_cnt)
    output_pol, last_cell_pol, memory_pol = decoder_lstm(embedded_decoder, initial_state=[input_pol_last_cell, input_pol_memory])
    pol_decoder = Model(inputs=[input_decoder, input_pol_last_cell, input_pol_memory], outputs=[output_pol, last_cell_pol, memory_pol])
    polish_embedding_model = Model(inputs=input_pol, outputs=embedded_polish)
    weights = (pol_int_out != wordInt_pol['\emptyChar']).astype(np.float64)
    
    epoch_cnt = 100
    batch_cnt = 256
    batch_size = 256
    for i in range(epoch_cnt):
        for batch in tqdm(range(batch_cnt)):
            batch_eng, batch_pol_in, batch_pol_out, sample_weights = make_batch(eng_ints, pol_int_in, pol_int_out, weights, batch_size)
            batch_pol_out = polish_embedding_model.predict(batch_pol_out)
            seq2seq.train_on_batch([batch_eng, batch_pol_in], batch_pol_out, sample_weight=sample_weights)

    embedded_polish_dict = embed(polish_embedding_model, intWord_pol)
    embedded_polish_dict = embedded_polish_dict.squeeze()

    eng_sentence_list = ['example sentence']
    eng_sentence_list = split_into_words(eng_sentence_list)
    eng_sentence_list = [sent[:max_eng_length] for sent in eng_sentence_list]
    eng_sentence_list = make_equal_length(eng_sentence_list, max_eng_length)
    eng_int = transform_to_ints(eng_sentence_list, wordInt_eng)
    eng_int = np.array(eng_int)
    last_cell, memory = eng_encoder.predict(eng_int)
    last_cell = last_cell.astype(np.float64)
    memory = memory.astype(np.float64)
    cnt = 0
    generated_word = '\start'
    result = []
    while cnt < max_pl_length and generated_word != '\end':
        dec_inp = np.array([[wordInt_pol[generated_word]]])
        generated_word, last_cell, memory = pol_decoder.predict([dec_inp, last_cell, memory])
        cnt += 1
        generated_word = vect2word(generated_word, embedded_polish_dict, intWord_pol)
        result.append(generated_word)
    print(" ".join(result))
