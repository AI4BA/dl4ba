import torch
import numpy as np
import argparse
import yaml
import utils
import json
from models import TextCNN, LSTM, MLP
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

seed=1234

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_name", type=str, default='word2vec', help="word2vec, glove, elmo, deepsim, bert")
    parser.add_argument("--dataset", type=str, default='gcc_data.csv', help="jdt_data_8.csv, gcc_data.csv, "
                                                                            "sun_firefox.csv)")
    parser.add_argument("--model", type=str, default='textcnn', help="textcnn, lstm, bi-lstm, mlp")
    parser.add_argument("--k_fold", type=int, default=10, help="k倍交叉验证")
    args = parser.parse_args()
    return args


def train_model(model, x_batch, y_batch, learning_rate, params, device):
    # 输入CNN
    # x_batch: batch_size, max_document_length, dimension
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    feature = torch.tensor(x_batch, dtype=torch.float).to(device)
    y_batch = np.array(y_batch)
    target = torch.tensor(y_batch, dtype=torch.float).to(device)
    score = model(feature)

    # 计算loss
    loss = F.cross_entropy(score, target)
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    loss += params["l2_reg_lambda"] * l2_reg

    loss.backward()
    optimizer.step()

    # 计算acc
    max_developer = torch.argmax(score, dim=1)
    # correct_pre: bool
    correct_predictions = torch.eq(max_developer, torch.argmax(target, dim=1))
    top1acc, top5acc, top10acc = utils.top_k_acc(correct_predictions, score, target)
    return loss, top1acc, top5acc, top10acc


def test_model(model, x_batch, y_batch, device):
    feature = torch.tensor(x_batch, dtype=torch.float)
    feature = feature.to(device)
    y_batch = np.array(y_batch)
    target = torch.tensor(y_batch, dtype=torch.float)
    target = target.to(device)
    feature = feature.to(device)
    score = model(feature)

    # 计算acc
    max_developer = torch.argmax(score, dim=1)
    # correct_pre: bool
    correct_predictions = torch.eq(max_developer, torch.argmax(target, dim=1))
    top1acc, top5acc, top10acc = utils.top_k_acc(correct_predictions, score, target)
    return top1acc, top5acc, top10acc


def operation(args):
    # load parameters
    with open("config.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    embedding_name = args.embedding_name
    embedding_dimension = config[embedding_name]['dimension']
    # load parameters for neural networks
    params = json.loads(open("parameters.json").read())
    # load datasets
    print("")
    print(
        "embedding_name:" + args.embedding_name + ", dataset:" + args.dataset + ", model:" + args.model + ", attention: " + str(
            params['attention']))
    print(
        "dropout:" + str(params['dropout_keep_prob']) + ", batch_size:" +str(params['batch_size']))
    filename = args.dataset
    text, y_onehot, _, labels = utils.load_data_and_labels(filename)
    spare = len(text) % 5
    rear = len(text)-spare
    text = text[:rear]
    y_onehot = y_onehot[:rear]
    x_tr, x_test, y_tr, y_test = train_test_split(text, y_onehot, test_size=0.2, random_state=seed)
    y_tr = np.array(y_tr)
    y_test = np.array(y_test)

    max_train_length = max([len(x.split(' ')) for x in x_tr])
    print('The maximum length of training text: {}'.format(max_train_length))

    max_test_length = max([len(x.split(' ')) for x in x_test])
    print('The maximum length of testing text: {}'.format(max_test_length))
    max_document_length = max_test_length if max_test_length >= max_train_length else max_train_length
    print('The maximum length of all text: {}'.format(max_document_length))

    with open('./labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    # vocab_size: 词表中词数
    # tokens: 文章转token list
    train_word_id_mat, train_vocab_size, train_word_to_id_dict, train_tokens = utils.vocabularyProcessor(
        max_document_length, x_tr)
    test_word_id_mat, test_vocab_size, test_word_to_id_dict, test_tokens = utils.vocabularyProcessor(
        max_document_length, x_test)
    # 得到词表的嵌入矩阵
    print(f"load {embedding_name} ...")
    if embedding_name == 'word2vec':
        bow = True
        utils.load_embedding_vectors_word2vec(train_word_to_id_dict, train_vocab_size, config, "train")
        utils.load_embedding_vectors_word2vec(test_word_to_id_dict, test_vocab_size, config, "test")
    elif embedding_name == 'glove':
        bow = True
        utils.load_embedding_vectors_glove(train_word_to_id_dict, train_vocab_size, config, "train")
        utils.load_embedding_vectors_glove(test_word_to_id_dict, test_vocab_size, config, "test")
    elif embedding_name == 'deepsim':
        bow = True
        utils.load_embedding_vectors_deepsim(train_word_to_id_dict, train_vocab_size, config, "train")
        utils.load_embedding_vectors_deepsim(test_word_to_id_dict, test_vocab_size, config, "test")
    elif embedding_name == 'elmo' or embedding_name == 'bert':
        bow = False

    print("")
    kernel_sizes = [int(k) for k in params["filter_sizes"].split(',')]

    print("Start training")
    # 进行训练和验证

    # 将数据分为十份
    num_epochs = params['num_epochs']
    k_fold = args.k_fold
    best_acc = 0
    best_k=0
    x_splits, y_splits = utils.data_split(y_tr, k_fold)

    # k倍交叉
    for i in range(k_fold):
        x_train = np.concatenate(x_splits[:i] + x_splits[i + 1:], axis=0)
        y_train = np.concatenate(y_splits[:i] + y_splits[i + 1:], axis=0)
        x_dev = x_splits[i]
        y_dev = y_splits[i]
    # 定义model
        model = None
        if args.model == "textcnn":
            model = TextCNN(embedding_dimension, y_tr.shape[1], len(kernel_sizes), kernel_sizes,
                            params['dropout_keep_prob'], params['attention'])
        elif args.model == "lstm":
            model = LSTM(embed_dim=embedding_dimension, class_num=y_tr.shape[1], bidirectional=False,
                            dropout=params['dropout_keep_prob'], attention=params['attention'])
        elif args.model == "bi-lstm":
            model = LSTM(embed_dim=embedding_dimension, class_num=y_tr.shape[1], bidirectional=True,
                            dropout=params['dropout_keep_prob'], attention=params['attention'])
        elif args.model == "mlp":
            model = MLP(input_dim=max_document_length * embedding_dimension, class_num=y_tr.shape[1],
                        attention=params['attention'])

        if torch.cuda.is_available():
            model = model.cuda()
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")



        # 迭代epoch次训练
        step = 0
        learning_rate = 0.001
        for epoch in range(num_epochs):
            # 分批次
            model.train()
            train_batches = utils.batch_iter(list(zip(x_train, y_train)), params['batch_size'])
            dev_batches = utils.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], params['num_epochs'])
            for train_batch in train_batches:
                step += 1
                x_train_batch, y_train_batch = zip(*train_batch)
                # 获得训练集的向量(batch_size, max_document_length, dimension)
                if bow:
                    input_x = utils.generate_text_embedding_bow(train_word_id_mat, x_train_batch, max_document_length,
                                                                embedding_dimension,
                                                                embedding_name, "train")
                else:
                    x_train_tokens = []
                    for paperid in x_train_batch:
                        x_train_tokens.append(train_tokens[paperid])
                    if embedding_name == 'bert':
                        input_x = utils.load_embedding_vectors_bert(max_document_length, x_train_tokens, config)
                    elif embedding_name == 'elmo':
                        input_x = utils.load_embedding_vectors_elmo(max_document_length, x_train_tokens, config)

                tloss, ttop1acc, ttop5acc, ttop10acc = train_model(model, input_x, y_train_batch, learning_rate, params,
                                                                   device)
                print(
                    f"epoch:{epoch + 1} train_step:{step} loss:{tloss:.6f} top1acc:{ttop1acc:.4f} top5acc:{ttop5acc:.4f} top10acc:{ttop10acc:.4f}")

        dev_step = 0
        model.eval()
        with torch.no_grad():
            for dev_batch in dev_batches:
                dev_step += 1
                x_dev_batch, y_dev_batch = zip(*dev_batch)
                if bow:
                    input_x = utils.generate_text_embedding_bow(train_word_id_mat, x_dev_batch, max_document_length,
                                                                embedding_dimension, embedding_name, "train")
                else:
                    x_dev_tokens = []
                    for paperid in x_dev_batch:
                        x_dev_tokens.append(train_tokens[paperid])
                    if embedding_name == 'bert':
                        input_x = utils.load_embedding_vectors_bert(max_document_length, x_dev_tokens, config)
                    elif embedding_name == 'elmo':
                            input_x = utils.load_embedding_vectors_elmo(max_document_length, x_dev_tokens, config)
                top1acc, top5acc, top10acc = test_model(model, input_x, y_dev_batch, device)
                print(
                    f"epoch:{epoch + 1} k:{i+1} dev_step:{dev_step} top1acc:{top1acc:.4f} top5acc:{top5acc:.4f} top10acc:{top10acc:.4f}")
                if top1acc > best_acc:
                    best_acc = top1acc
                    best_k=i+1
                    torch.save(model,
                                'trained_models/' + args.model + '_' + args.embedding_name + '_' + args.dataset + '_' + str(
                                    params['batch_size']) + '_' + str(
                                    params['dropout_keep_prob']) + str(params['attention']) + '.pth')

        print("")
    print("")
    print(f"best:top1acc:{best_acc} k:{best_k} ")
    print("------------------------")
    print("------------------------")

    # 开始测试
    print("Start testing")
    model = torch.load('trained_models/' + args.model + '_' + args.embedding_name + '_' + args.dataset + '_' + str(
        params['batch_size']) + '_' + str(
        params['dropout_keep_prob']) + str(params['attention']) + '.pth')
    model.eval()
    # 分批次
    x_splits, y_splits = utils.data_split(y_test, 1)
    x_te = x_splits[0]
    y_te = y_splits[0]
    test_batches = utils.batch_iter(list(zip(x_te, y_te)), params['batch_size'])
    step = 0
    for test_batch in test_batches:
        step += 1
        x_test_batch, y_test_batch = zip(*test_batch)
        if bow:
            input_x = utils.generate_text_embedding_bow(test_word_id_mat, x_test_batch, max_document_length,
                                                        embedding_dimension, embedding_name, "test")
        else:
            x_test_tokens = []
            for paperid in x_test_batch:
                x_test_tokens.append(test_tokens[paperid])
            if embedding_name == 'bert':
                input_x = utils.load_embedding_vectors_bert(max_document_length, x_test_tokens, config)
            elif embedding_name == 'elmo':
                        input_x = utils.load_embedding_vectors_elmo(max_document_length, x_test_tokens, config)
        ttop1acc, ttop5acc, ttop10acc = test_model(model, input_x, y_test_batch, device)
        print(
            f"epoch:t test_step:{step} top1acc:{ttop1acc:.4f} top5acc:{ttop5acc:.4f} top10acc:{ttop10acc:.4f}")


if __name__ == '__main__':
    args = parse_args()
    operation(args)
    print("Done")
