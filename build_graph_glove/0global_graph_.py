# -*- coding: utf-8 -*-
# relevant sentences are selected by topk, the corresponding weights are got using bert
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

def global_adj_matrix(sentence, relevant_sentences_index, tf_idf_dic):
    relevant_sentences_index = relevant_sentences_index.strip().split()
    # relevant_sentences_value = relevant_sentences_value.strip().split()
    matrix = np.zeros((len(sentence), len(relevant_sentences_index))).astype('float32')
    
    for i in range(len(sentence)):
        for j in range(len(relevant_sentences_index)):
            try:
                matrix[i][j] = float(tf_idf_dic[sentence[i]][int(relevant_sentences_index[j])])
            except IndexError:
                import pdb;pdb.set_trace()
            # matrix[i][j] = float(relevant_sentences_value[j])
    return matrix

def process(dataset):
    print("--------read data!-------")
    # read data
    train_data = open('../ASTE-Data-V2/'+dataset+'/train_triplets.txt','r').read().strip('\n').split('\n')
    test_data = open('../ASTE-Data-V2/'+dataset+'/test_triplets.txt','r').read().strip('\n').split('\n')
    dev_data = open('../ASTE-Data-V2/'+dataset+'/dev_triplets.txt','r').read().strip('\n').split('\n')
    print('train, dev, test!')
    train_sentences, dev_sentences, test_sentences = \
        [line.split('####')[0].split() for line in train_data], [line.split('####')[0].split() for line in dev_data], [line.split('####')[0].split() for line in test_data]
    all_sentences = train_sentences + dev_sentences + test_sentences
    # store graphs
    idx2graph_train, idx2graph_dev, idx2graph_test = {}, {}, {}
    # save graph
    fout_train = open('../ASTE-Graph-V2/'+dataset+'/global_graph0/train_g_final.graph', 'wb')
    fout_dev = open('../ASTE-Graph-V2/'+dataset+'/global_graph0/dev_g_final.graph', 'wb')
    fout_test = open('../ASTE-Graph-V2/'+dataset+'/global_graph0/test_g_final.graph', 'wb')
    # load graph edges
    # tf_idf_dic = load_dict('../tf_idf/'+dataset+'_tf_idf')
    print('--------load edges!--------')
    tf_idf_dic = load_dict('../edge_weights/'+dataset+'_word_sentence_edge')
    # load relevant sentences
    print('--------relevant sentences!--------')
    train_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/train_r_fine_tune_52.txt', 'r').read().split('\n')
    dev_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/dev_r_fine_tune_52.txt', 'r').read().split('\n')
    test_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/test_r_fine_tune_52.txt', 'r').read().split('\n')
    # build graph
    for i in tqdm(range(len(train_sentences))):
        idx2graph_train[i] = global_adj_matrix(train_sentences[i], train_relevant_sentences[i], tf_idf_dic)
    pkl.dump(idx2graph_train, fout_train)

    for i in tqdm(range(len(dev_sentences))):
        idx2graph_dev[i] = global_adj_matrix(dev_sentences[i], dev_relevant_sentences[i], tf_idf_dic)
    pkl.dump(idx2graph_dev, fout_dev)

    for i in tqdm(range(len(test_sentences))):
        idx2graph_test[i] = global_adj_matrix(test_sentences[i], test_relevant_sentences[i], tf_idf_dic)
    pkl.dump(idx2graph_test, fout_test)

    fout_train.close()
    fout_dev.close()
    fout_test.close()

if __name__ == '__main__':
    process('res14')
    process('res15')
    process('lap14')
    process('res16')