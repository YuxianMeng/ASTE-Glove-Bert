# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle as pkl
from tqdm import tqdm
from spacy.tokens import Doc

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

def local_adj_matrix(sentence, pmi_dic):
    matrix = np.zeros((len(sentence), len(sentence))).astype('float32')

    for i in range(len(sentence)):
        for j in range(len(sentence)):
            word1 = sentence[i]
            word2 = sentence[j]
            if word1 != word2:
                weight = pmi_dic[word1+','+word2]
                matrix[i][j] = weight
    # for triple in triple_list:
    #         for i in triple[0]:
    #             for j in triple[1]:
    #                 matrix[i][j] = triple[2]

    return matrix

def process(dataset):
    print("--------read data!-------")

    train_data = open('../ASTE-Data/'+dataset+'/train.txt','r').read().split('\n')[:-1]
    test_data = open('../ASTE-Data/'+dataset+'/test.txt','r').read().split('\n')[:-1]
    dev_data = open('../ASTE-Data/'+dataset+'/dev.txt','r').read().split('\n')[:-1]
    all_data = train_data + test_data + dev_data
    
    idx2graph_train, idx2graph_dev, idx2graph_test = {}, {}, {}

    fout_train = open('../ASTE-Data/'+dataset+'/local_graph/train_l4.graph', 'wb')
    fout_dev = open('../ASTE-Data/'+dataset+'/local_graph/dev_l4.graph', 'wb')
    fout_test = open('../ASTE-Data/'+dataset+'/local_graph/test_l4.graph', 'wb')

    train_sentences, test_sentences, dev_sentences = [], [], []

    print('load word word edge')
    pmi_dic = load_dict('../edge_weights/'+dataset+'_word_word_edge')

    for line in tqdm(train_data):
        train_sentences.append(line.split('####')[0].strip().split())

    for line in tqdm(test_data):
        test_sentences.append(line.split('####')[0].strip().split())

    for line in tqdm(dev_data):
        dev_sentences.append(line.split('####')[0].strip().split())

    all_sentences = train_sentences + test_sentences + dev_sentences

    for i in range(len(train_sentences)):
        idx2graph_train[i] = local_adj_matrix(train_sentences[i], pmi_dic)
    pkl.dump(idx2graph_train, fout_train)

    for i in range(len(dev_sentences)):
        idx2graph_dev[i] = local_adj_matrix(dev_sentences[i], pmi_dic)
    pkl.dump(idx2graph_dev, fout_dev)

    for i in range(len(test_sentences)):
        idx2graph_test[i] = local_adj_matrix(test_sentences[i], pmi_dic)
    pkl.dump(idx2graph_test, fout_test)
    # import pdb; pdb.set_trace()
    fout_train.close()
    fout_dev.close()
    fout_test.close()

if __name__ == '__main__':
    process('res14')
    process('res15')
    process('lap14')
    process('res16')