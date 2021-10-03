# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle as pkl
from tqdm import tqdm
from spacy.tokens import Doc
import pdb

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

# def load_dict(name ):
#     return np.load(name + '.npy', allow_pickle='TRUE')

def global_adj_matrix(relevant_sentences_index, tf_idf_list, all_sentences):
    # data prepare
    relevant_sentences_index = relevant_sentences_index.strip().split()
    # init zero matrix
    matrix = []
    # i is the index of column relevant sentence
    for i in range(len(relevant_sentences_index)):
        # prepare
        sentence_col_index = int(relevant_sentences_index[i])
        sentence_col = all_sentences[sentence_col_index]
        sentence_word_col = sentence_col
        # matrix for relevant sentence i
        tem_matrix = np.zeros((len(relevant_sentences_index), len(sentence_word_col))).astype('float32')
        # j is the index of row relevant sentence
        for j in range(len(relevant_sentences_index)):
            for k in range(len(sentence_word_col)):
                sentence_row_index = int(relevant_sentences_index[j])
                word = sentence_word_col[k]
                weight = tf_idf_list[word][sentence_row_index]
                tem_matrix[j][k] = weight.item()
        matrix.append(tem_matrix)
    return matrix

def process(dataset):
    print("--------read data!-------")
    # read data
    train_data = open('../ASTE-Data-V2/'+dataset+'/train_triplets.txt','r').read().strip('\n').split('\n')
    test_data = open('../ASTE-Data-V2/'+dataset+'/test_triplets.txt','r').read().strip('\n').split('\n')
    dev_data = open('../ASTE-Data-V2/'+dataset+'/dev_triplets.txt','r').read().strip('\n').split('\n')
    train_sentences, dev_sentences, test_sentences = \
        [line.split('####')[0].split() for line in train_data], [line.split('####')[0].split() for line in dev_data], [line.split('####')[0].split() for line in test_data]
    all_sentences = train_sentences + dev_sentences + test_sentences
    print(len(train_data), len(dev_data), len(test_data))
    # store graphs
    idx2graph_train, idx2graph_dev, idx2graph_test = {}, {}, {}
    # file 
    fout_train = open('../ASTE-Graph-V2/'+dataset+'/global_graph1/train_g_final.graph', 'wb')
    fout_dev = open('../ASTE-Graph-V2/'+dataset+'/global_graph1/dev_g_final.graph', 'wb')
    fout_test = open('../ASTE-Graph-V2/'+dataset+'/global_graph1/test_g_final.graph', 'wb')
    # relevant sentences
    train_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/train_r_fine_tune_52.txt', 'r').read().split('\n')
    dev_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/dev_r_fine_tune_52.txt', 'r').read().split('\n')
    test_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/test_r_fine_tune_52.txt', 'r').read().split('\n')
    print(len(train_relevant_sentences), len(dev_relevant_sentences), len(test_relevant_sentences))
    # tf_idf_dic = load_dict('../tf_idf/'+dataset+'_tf_idf')
    tf_idf_dic = load_dict('../edge_weights/'+dataset+'_word_sentence_edge')
    # build graph
    for i in range(len(train_sentences)):
        idx2graph_train[i] = global_adj_matrix(train_relevant_sentences[i], tf_idf_dic, all_sentences)
    pkl.dump(idx2graph_train, fout_train)

    for i in range(len(dev_sentences)):
        try:
            idx2graph_dev[i] = global_adj_matrix(dev_relevant_sentences[i], tf_idf_dic, all_sentences)
        except IndexError:
            import pdb; pdb.set_trace()
    pkl.dump(idx2graph_dev, fout_dev)

    for i in range(len(test_sentences)):
        idx2graph_test[i] = global_adj_matrix(test_relevant_sentences[i], tf_idf_dic, all_sentences)
    pkl.dump(idx2graph_test, fout_test)

    fout_train.close()
    fout_dev.close()
    fout_test.close()

if __name__ == '__main__':
    process('res14')
    process('res15')
    process('lap14')
    process('res16')