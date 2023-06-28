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


def load_data_and_labels(filename):
    """
    filename = 'xxx.csv.zip'
    x_raw: 清洗后的description
    y_raw: labels对应的one-hot矩阵
    df: dataset
    labels: ['Lars.Vogel', 'Michael_Rennie', ...]
    """
    print("")
    print("load data ...")
    dataset = pd.read_csv(filename)
    if filename == 'gcc_data.csv':
        selected = ['Assignee', 'Summary', 'Description']
    else:
        selected = ['Assignee', 'Description']
    non_selected = list(set(dataset.columns) - set(selected))
    dataset = dataset.drop(non_selected, axis=1)  # delete non selected columns
    dataset = dataset.dropna(axis=0, how='any', subset=selected)  # delete null rows
    # dataset = dataset.reindex(np.random.permutation(dataset.index))
    labels = sorted(list(set(dataset[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    if filename == 'gcc_data.csv':
        dataset['merge'] = dataset.Summary + ' ' + dataset.Description
        x = dataset['merge'].apply(lambda x: clean_str(x)).tolist()
    else:
        x = dataset[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y = dataset[selected[0]].apply(lambda y: label_dict[y]).tolist()
    return x, y, dataset, labels


def vocabularyProcessor(max_document_length, texts):
    # Tokenize text
    tokens = [sentence.split() for sentence in texts]
    # Count word frequency
    word_counts = Counter(word for sentence in tokens for word in sentence)
    word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    # Create word to id mapping
    word_to_id_dict = {word: i + 1 for i, (word, count) in enumerate(word_counts)}
    # Convert tokens to ids
    vocab = [[word_to_id_dict[word] for word in sentence] for sentence in tokens]
    # Pad sequences
    vocab = [
        sentence + [0] * (max_document_length - len(sentence)) if len(sentence) < max_document_length else sentence[
                                                                                                           :max_document_length]
        for sentence in vocab]
    vocab = np.array(vocab)
    # Get vocab size
    vocab_size = len(word_to_id_dict) + 1
    return vocab, vocab_size, word_to_id_dict, tokens


def batch_iter(data, batch_size, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data, dtype=object)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def generate_text_embedding_bow(word_id_mat, x_batch, max_document_length, embedding_dim, embed_name, state):
    # x_batch：文章号
    # y_batch：one-hot
    # max_document_length: 最大长度-padding
    W = None
    input_x = np.random.uniform(-0.25, 0.25, (len(x_batch), max_document_length, embedding_dim))
    if embed_name == 'word2vec':
        W = np.loadtxt("embeddings/"+state+"_word2vec_embedding_vectors.csv", delimiter=',')
    elif embed_name == 'glove':
        W = np.loadtxt("embeddings/"+state+"_glove_embedding_vectors.csv", delimiter=',')
    elif embed_name == 'deepsim':
        W = np.loadtxt("embeddings/"+state+"_deepsim_embedding_vectors.csv", delimiter=',')
    if embed_name == 'elmo':
        W = np.loadtxt("embeddings/"+state+"_elmo_embedding_vectors.csv", delimiter=',')
    for i, paper_NO in enumerate(x_batch):
        for j, word_id in enumerate(word_id_mat[paper_NO]):
            input_x[i, j, :] = W[word_id]
    return input_x


def top_k_acc(correct_predictions, score, target):
    # top 1 acc
    top_1_acc = torch.mean(correct_predictions.float())

    # top 5 acc
    # sort
    _, top_5_developers = torch.topk(score, 5)
    top_5_developers = top_5_developers.t()
    top_5_correct_predictions = top_5_developers.eq(torch.argmax(target, dim=1).expand_as(top_5_developers))
    top_5_correct_predictions = torch.any(top_5_correct_predictions, dim=0)
    top_5_acc = torch.mean(top_5_correct_predictions.float())

    # top 10 acc
    _, top_10_developers = torch.topk(score, 10)
    top_10_developers = top_10_developers.t()
    top_10_correct_predictions = top_10_developers.eq(torch.argmax(target, dim=1).expand_as(top_10_developers))
    top_10_correct_predictions = torch.any(top_10_correct_predictions, dim=0)
    top_10_acc = torch.mean(top_10_correct_predictions.float())
    return top_1_acc, top_5_acc, top_10_acc


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


def data_split(y, k_fold):
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = shuffle_indices
    y_shuffled = y[shuffle_indices]
    x_splits = np.array_split(x_shuffled, k_fold)
    y_splits = np.array_split(y_shuffled, k_fold)
    return x_splits, y_splits
