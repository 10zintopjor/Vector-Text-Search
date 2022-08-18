import encodings
from lib2to3.pytree import convert
from re import search
from sqlite3 import converters
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
from botok import WordTokenizer
import faiss
import pyewts
from pathlib import Path


trgt_txt = Path("target_file.txt").read_text(encoding="utf-8")
word_vectors_path = "word2vec.wordvectors"
wv = KeyedVectors.load(str(word_vectors_path), mmap='r')
wt = WordTokenizer()
converter = pyewts.pyewts()
window = 5

def create_index():
    vectors = []
    wylie_text = toWylie(trgt_txt)
    words = wt.tokenize(wylie_text)
    for word in words:
        vectors.append(get_vector(wv,word.lemma))
    sum_vectors = np.asarray(get_sum_vectors(vectors)).astype("float32")
    index = faiss.IndexFlatL2(100)
    index.add(sum_vectors)
    return index

    

def get_sum_vectors(vectors):
    sum_vectors = []
    for i in range(0,len(vectors),window):
        k = i+window
        sum_vectors.append(np.mean(vectors[i:k],axis=0))
    return sum_vectors


def get_vector(vector_model,word):
    if word in vector_model.key_to_index.keys():
        return vector_model.get_vector(word)
    else:
        return np.zeros(vector_model.vector_size)

def search(index):
    beg_vectors = []
    end_vectors = []
    trgt_txt = Path("search_file.txt").read_text(encoding="utf-8")
    words = wt.tokenize(trgt_txt)
    beg_words = ""
    end_words = ""
    for word in words[:window]:
        beg_words+=word.text
        beg_vectors.append(get_vector(wv,word.lemma))

    for word in words[-window:]:
        end_words+=word.text
        end_vectors.append(get_vector(wv,word.lemma))

    beg_vector = np.mean(beg_vectors,axis=0)
    end_vector = np.mean(end_vectors,axis=0)
    search_result = index.search(np.array([beg_vector, end_vector]).astype("float32"), 1)
    print(search_result[1][0][0])
    print(search_result[1][1][0])
    print(beg_words)
    print(end_words)

    return search_result[1][0][0],search_result[1][1][0]


def test_vector(beg_index,end_index):
    vectors = []
    words = wt.tokenize(trgt_txt)
    for word in words:
        vectors.append(get_vector(wv,word.lemma))
    j=0
    for i in range(0,len(vectors),window):
        k = i+window
        check_vector = np.mean(vectors[i:k],axis=0)  
        if j == beg_index.item():
            test_txt = ""
            for word in words[i:k]:
                test_txt+=word.text
            print(test_txt)    
        j+=1         



def toWylie(unicode_text):
    wylie_text = converter.toWylie(unicode_text)
    return wylie_text

def toUnicode(wylie_text):
    unicode_text = converter.toUnicode(wylie_text)
    return unicode_text


index = create_index()
beg_index,end_index = search(index)

test_vector(beg_index,end_index)