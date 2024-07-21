from translate.storage.tmx import tmxfile

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

if __name__ == '__main__':
    tmx_file = read_tmx_file()
    examples = list(tmx_file.unit_iter())
    eng = [ex.source for ex in examples]
    pol = [ex.target for ex in examples]
