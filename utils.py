import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import torch
from collections import Counter
from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertTokenizer, BertModel


def clean_str(s):
    s = re.sub(r"[^A-Za-z(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " had", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " ", s)
    s = re.sub(r"\)", " ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*', "xx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    tokens = s.split(' ')
    data = []
    words = stopwords.words('english')
    for token in tokens:
        if token not in words and len(token)>1:
            data.append(token)
    s = " ".join(data)
    return s.strip().lower()


def load_embedding_vectors_word2vec(vocabulary, embed_size, config, state):
    embed_model = config['word2vec']['path']
    embed_dim = config['word2vec']['dimension']
    with open(embed_model, "rb") as f:
        header = f.readline()
        vocab_size, _ = map(int, header.split())

        embedding_vectors = np.random.uniform(-0.25, 0.25, (embed_size, embed_dim))
        binary_len = np.dtype('float32').itemsize * embed_dim
        for line_no in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b'\n':
                    word.append(ch)
            word = b''.join(word).decode("utf-8")
            if word in vocabulary:
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.seek(binary_len, 1)
        f.close()
    with open('embeddings/'+ state+'_word2vec_embedding_vectors.csv', 'w') as f:
        np.savetxt(f, embedding_vectors, fmt='%f', delimiter=',')


def load_embedding_vectors_glove(vocabulary, embed_size, config, state):
    embed_model = config['glove']['path']
    embed_dim = config['glove']['dimension']
    embedding_vectors = np.random.uniform(-0.25, 0.25, (embed_size, embed_dim))
    f = open(embed_model)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        if word in vocabulary:
            idx = vocabulary.get(word)
            if idx != 0:
                embedding_vectors[idx] = vector
    f.close()
    with open('embeddings/'+state+'_glove_embedding_vectors.csv', 'w') as f:
        np.savetxt(f, embedding_vectors, fmt='%f', delimiter=',')


def load_embedding_vectors_deepsim(vocabulary, embed_size, config, state):
    embed_model = config['deepsim']['path']
    embed_dim = config['deepsim']['dimension']
    embedding_vectors = np.random.uniform(-0.25, 0.25, (embed_size, embed_dim))
    with open(embed_model, "rb") as f:
        header = f.readline()
        vocab_size, _ = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * embed_dim
        for line_no in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b'\n':
                    word.append(ch)
            word = b''.join(word).decode("utf-8")
            if word in vocabulary:
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.seek(binary_len, 1)
        f.close()
    with open('embeddings/'+state+'_deepsim_embedding_vectors.csv', 'w') as f:
        np.savetxt(f, embedding_vectors, fmt='%f', delimiter=',')


def load_embedding_vectors_elmo(max_document_length, tokens, config):
    dimension = config['elmo']['dimension']
    options_file = config['elmo']['options_file']
    weight_file = config['elmo']['weight_file']
    num_papers = len(tokens)
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    shape = (num_papers, max_document_length, dimension)
    embedding_vectors = torch.FloatTensor(*shape).uniform_()

    input_ids = batch_to_ids(tokens)
    embeddings = elmo(input_ids)['elmo_representations'][0]
    embedding_vectors[:, :embeddings.shape[1], :] = embeddings
    embedding_vectors = embedding_vectors.detach().numpy()
    return embedding_vectors


def load_embedding_vectors_bert(max_document_length, tokens, config):
    max_token_length = config['bert']['max_token_length']
    dimension = config['bert']['dimension']
    num_papers = len(tokens)
    tokenizer = BertTokenizer.from_pretrained(config['bert']['tokenizer'])
    model = BertModel.from_pretrained(config['bert']['path'])
    shape = (num_papers, max_document_length, dimension)
    embedding_vectors = torch.FloatTensor(*shape).uniform_()

    for paperNo, token in enumerate(tokens):
        paper_size = len(token)
        embedding_paper = torch.zeros(paper_size, dimension)

        # 转token_id
        input_ids = tokenizer.convert_tokens_to_ids(token)
        # 分段
        start = 0
        while start < paper_size:
            if start + max_token_length >= paper_size:
                end = paper_size
            else:
                end = start + max_token_length
            seg = torch.tensor([input_ids[start:end]])
            outputs = model(seg)
            embedding_per_seg = outputs[0][0]
            embedding_paper[start:end] = embedding_per_seg
            start += max_token_length
        embedding_vectors[paperNo, :paper_size, :] = embedding_paper
    embedding_vectors = embedding_vectors.detach().numpy()
    return embedding_vectors

