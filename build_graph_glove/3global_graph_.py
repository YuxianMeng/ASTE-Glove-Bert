# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle as pkl
from tqdm import tqdm
from spacy.tokens import Doc
import pdb
from transformers import BertModel, BertTokenizer
import torch.nn as nn

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

# def load_dict(name ):
#     return np.load(name + '.npy', allow_pickle='TRUE')

def global_adj_matrix(sentence, relevant_sentences_index, tokenizer, all_sentences, bert_model, pmi_dic):
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
        tem_matrix = np.zeros((len(sentence), len(sentence_word_col))).astype('float32')
        # j is the index of row relevant sentence
        for j in range(len(sentence)):
            for k in range(len(sentence_word_col)):
                word_row_in_s0 = sentence[j]
                word_col_in_sk = sentence_word_col[k]
                # using bert to get the representation of two words
                # input_s0 = tokenizer(word_row_in_s0, max_length=5, truncation=True, padding='max_length', return_tensors='pt')
                # s0_representation = bert_model(input_s0.input_ids.to('cuda'), input_s0.attention_mask.to('cuda'))[0][0][0]
                # input_sk = tokenizer(word_col_in_sk, max_length=5, truncation=True, padding='max_length', return_tensors='pt')
                # sk_representation = bert_model(input_sk.input_ids.to('cuda'), input_sk.attention_mask.to('cuda'))[0][0][0]
                # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                # weight = cos(s0_representation, sk_representation)
                # using pmi
                # weight = 0
                # if word_row_in_s0 == word_col_in_sk:
                #     weight = 1
                # else:
                word_pair = str(word_row_in_s0)+','+str(word_col_in_sk)
                if word_pair in pmi_dic.keys():
                    weight = pmi_dic[word_pair]
                    tem_matrix[j][k] = weight
        matrix.append(tem_matrix)
    return matrix

def process(dataset, pretrained='bert-base-uncased'):
    print("--------read data!-------")

    train_data = open('../ASTE-Data-V2/'+dataset+'/train_triplets.txt','r').read().strip('\n').split('\n')
    test_data = open('../ASTE-Data-V2/'+dataset+'/test_triplets.txt','r').read().strip('\n').split('\n')
    dev_data = open('../ASTE-Data-V2/'+dataset+'/dev_triplets.txt','r').read().strip('\n').split('\n')
    
    idx2graph_train, idx2graph_dev, idx2graph_test = {}, {}, {}

    fout_train = open('../ASTE-Graph-V2/'+dataset+'/global_graph3/train_g_final.graph', 'wb')
    fout_dev = open('../ASTE-Graph-V2/'+dataset+'/global_graph3/dev_g_final.graph', 'wb')
    fout_test = open('../ASTE-Graph-V2/'+dataset+'/global_graph3/test_g_final.graph', 'wb')

    train_sentences, test_sentences, dev_sentences = [], [], []
    train_sentences, dev_sentences, test_sentences = \
        [line.split('####')[0].split() for line in train_data], [line.split('####')[0].split() for line in dev_data], [line.split('####')[0].split() for line in test_data]

    # bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    bert_model = BertModel.from_pretrained(pretrained).to('cuda')

    all_sentences = train_sentences + dev_sentences + test_sentences

    # tf_idf_dic = load_dict('../tf_idf/'+dataset+'_tf_idf')
    pmi_dic = load_dict('../edge_weights/'+dataset+'_word_word_edge')

    train_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/train_r_fine_tune_52.txt', 'r').read().split('\n')
    dev_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/dev_r_fine_tune_52.txt', 'r').read().split('\n')
    test_relevant_sentences = open('../ASTE-Rele-Sentences/'+dataset+'/test_r_fine_tune_52.txt', 'r').read().split('\n')

    for i in tqdm(range(len(train_sentences))):
        idx2graph_train[i] = global_adj_matrix(train_sentences[i], train_relevant_sentences[i], tokenizer, all_sentences, bert_model, pmi_dic)
    pkl.dump(idx2graph_train, fout_train)

    for i in tqdm(range(len(dev_sentences))):
        idx2graph_dev[i] = global_adj_matrix(dev_sentences[i], dev_relevant_sentences[i], tokenizer, all_sentences, bert_model, pmi_dic)
    pkl.dump(idx2graph_dev, fout_dev)

    for i in tqdm(range(len(test_sentences))):
        idx2graph_test[i] = global_adj_matrix(test_sentences[i], test_relevant_sentences[i], tokenizer, all_sentences, bert_model, pmi_dic)
    pkl.dump(idx2graph_test, fout_test)

    fout_train.close()
    fout_dev.close()
    fout_test.close()

if __name__ == '__main__':
    process('res14')
    process('res15')
    process('lap14')
    process('res16')