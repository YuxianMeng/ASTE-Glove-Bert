# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pdb
from tqdm import tqdm

def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ABSADatasetReader:
    @staticmethod
    def __read_text__(fnames):
        '''a string: sentence1\nsentence2\n...sentencen\n'''
        text = ''
        for fname in fnames:
            fin = open(fname)
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines)):
                text += lines[i].split('####')[0].lower().strip()+'\n'
        return text

    @staticmethod
    def __read_triplets__(fnames):
        '''a list: [[([2], [5], 'NEG')], [(),()], [], ..., []]'''
        triplets = []
        for fname in fnames:
            fin = open(fname) 
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines)):
                triple = eval(lines[i].split('####')[1])
                triplets.append(triple)
        return triplets

    @staticmethod
    def __read_all_sentence__(domain):
        '''read all sentence (train/dev/test) to get the representation of relevant sentences'''
        train_data = open('./ASTE-Data-V2/'+domain+'/train_triplets.txt','r').readlines()
        dev_data = open('./ASTE-Data-V2/'+domain+'/dev_triplets.txt','r').readlines()
        test_data = open('./ASTE-Data-V2/'+domain+'/test_triplets.txt','r').readlines()

        train_sentences = [line.split('####')[0] for line in train_data]
        dev_sentences = [line.split('####')[0] for line in dev_data]
        test_sentences = [line.split('####')[0] for line in test_data]
        all_sentences = train_sentences + dev_sentences + test_sentences
        return all_sentences

    @staticmethod
    def __triple2bio__(sentences, triplets):
        '''
            convert triplets to BIO labels
            000120000000
            000000012220
            000330000000
            pos1, neg2, neu3
        '''
        sentences = sentences.strip('\n').split('\n')
        sentiment_dic = {'POS':1, 'NEG':2, 'NEU':3}
        aspect_labels, opinion_labels, sentiment_labels = [], [], []
        for sentence, triplet in zip(sentences, triplets):
            sentence = sentence.strip('\n').split()
            a_labels = [0 for i in range(len(sentence))]
            o_labels = [0 for i in range(len(sentence))]
            s_labels = [0 for i in range(len(sentence))]
            for tri in triplet:
                begin, inside = 1, 2
                a_index, o_index, polarity = tri
                for i in range(len(a_index)):
                    if i == 0:
                        a_labels[a_index[i]] = begin
                        s_labels[a_index[i]] = sentiment_dic[polarity]
                    else:
                        a_labels[a_index[i]] = inside
                        s_labels[a_index[i]] = sentiment_dic[polarity]
                for i in range(len(o_index)):
                    if i == 0:
                        o_labels[o_index[i]] = begin
                    else:
                        o_labels[o_index[i]] = inside
            aspect_labels.append(a_labels)
            opinion_labels.append(o_labels)
            sentiment_labels.append(s_labels)
        return aspect_labels, opinion_labels, sentiment_labels

    @staticmethod
    def __triple2span__(sentences, triplets):
        ''' 
            convert bio labels to span labels
            00000
            01000
            00000
            00000 
            the index of 1 denotes the start and end of term
        '''
        sentences = sentences.strip('\n').split('\n')
        aspect_span, opinion_span = [], []
        for sentence, triple in zip(sentences, triplets):
            sentence = sentence.strip('\n').split()
            matrix_span_aspect = np.zeros((len(sentence), len(sentence))).astype('float32')
            matrix_span_opinion = np.zeros((len(sentence), len(sentence))).astype('float32')
            for tri in triple:
                a_start, a_end, o_start, o_end = tri[0][0], tri[0][-1], tri[1][0], tri[1][-1]
                matrix_span_aspect[a_start][a_end] = 1
                matrix_span_opinion[o_start][o_end] = 1
            aspect_span.append(matrix_span_aspect)
            opinion_span.append(matrix_span_opinion)
        return aspect_span, opinion_span

    @staticmethod
    def __triple2grid__(sentences, triplets):
        ''' 
            convert triplets to grid label for pair and triplet
            row aspect, col opinion
            00000  00000
            01220  03330
            02220  03330 pos1 neg2 neu3 
            padding
        '''
        sentiment_dic = {'POS':1, 'NEG':2, 'NEU':3}
        sentences = sentences.strip('\n').split('\n')
        pair_grid_labels, triple_grid_labels = {}, {}
        for i in range(len(sentences)):
            sentence, triplet = sentences[i].strip('\n').split(), triplets[i]
            matrix_pair = np.zeros((len(sentence), len(sentence))).astype('float32')
            matrix_triple = np.zeros((len(sentence), len(sentence))).astype('float32')
            for tri in triplet:
                for j in tri[0]:
                    for k in tri[1]:
                        matrix_pair[j][k] = 2
                        matrix_triple[j][k] = sentiment_dic[tri[2]]
                matrix_pair[tri[0][0]][tri[1][0]] = 1
            pair_grid_labels[i] = matrix_pair
            triple_grid_labels[i] = matrix_triple
        return pair_grid_labels, triple_grid_labels

    @staticmethod
    def __mask__(sentences):
        sentences = sentences.strip('\n').split('\n')
        mask = []
        for sentence in sentences:
            sentence = sentence.strip('\n').split()
            mask.append([1]*len(sentence))
        return mask

    @staticmethod
    def __read_data__(fname, domain, phase, tokenizer):
        # read raw data
        sentence = ABSADatasetReader.__read_text__([fname]) # a long string splited by '\n'
        triplets = ABSADatasetReader.__read_triplets__([fname]) # a long list containing multiple lists for sentences 
        assert len(sentence.strip('\n').split('\n')) == len(triplets)
        all_sentences = ABSADatasetReader.__read_all_sentence__(domain)
        # generate basic labels
        aspect_sequence_labels, opinion_sequence_labels, sentiment_sequence_labels = ABSADatasetReader.__triple2bio__(sentence, triplets)
        aspect_span_labels, opinion_span_labels = ABSADatasetReader.__triple2span__(sentence, triplets)
        pair_grid_labels, triple_grid_labels = ABSADatasetReader.__triple2grid__(sentence, triplets)
        text_mask = ABSADatasetReader.__mask__(sentence)
        # read relevant sentences 
        relevant_sentences_index = open('./ASTE-Rele-Sentences/'+domain + '/' + phase + '_r_fine_tune_52.txt', 'r').read().split('\n')
        # local graph
        # local_graph = pickle.load(open('./ASTE-Graph-V2/' + domain + '/local_graph/' + phase + '_l.graph', 'rb'))
        # four types of global graphs
        global_graph0 = pickle.load(open('./ASTE-Graph-V2/' + domain + '/global_graph0/' + phase + '_g_final.graph', 'rb'))
        global_graph1 = pickle.load(open('./ASTE-Graph-V2/' + domain + '/global_graph1/' + phase + '_g_final.graph', 'rb'))
        global_graph2 = pickle.load(open('./ASTE-Graph-V2/' + domain + '/global_graph2/' + phase + '_g_final.graph', 'rb'))
        global_graph3 = pickle.load(open('./ASTE-Graph-V2/' + domain + '/global_graph3/' + phase + '_g_final.graph', 'rb'))
        # store all data for bucket
        all_data = []
        lines = sentence.strip('\n').split('\n')
        for i in range(0, len(lines)):
            # raw text, text indices and text mask
            text = lines[i].lower().strip()
            text_indices = tokenizer.text_to_sequence(text)
            mask = text_mask[i]
            # index of relevant sentence for this sentence
            relevant_sentences = [int(idx) for idx in relevant_sentences_index[i].strip().split()]
            # indieces of relevant sentence for this sentence (representation)
            relevant_sentences_presentation = []
            for mm in relevant_sentences:
                tem_sentence = all_sentences[mm]
                sentence_indices = tokenizer.text_to_sequence(tem_sentence)
                relevant_sentences_presentation.append(sentence_indices)    
            # different graphs for this sentence
            # local_graph_ = local_graph[i]
            global_graph_0, global_graph_1, global_graph_2, global_graph_3 = \
                global_graph0[i], global_graph1[i], global_graph2[i], global_graph3[i]
            # different labels for this sentence
            aspect_sequence_label, opinion_sequence_label, sentiment_sequence_label, aspect_span_label, opinion_span_label, pair_grid_label, triple_grid_label = \
                aspect_sequence_labels[i], opinion_sequence_labels[i], sentiment_sequence_labels[i], aspect_span_labels[i], opinion_span_labels[i], \
                    pair_grid_labels[i], triple_grid_labels[i]  
            # package
            data = {
                'text_indices': text_indices,
                'mask': mask,
                'global_graph0': global_graph_0,
                'global_graph1': global_graph_1,
                'global_graph2': global_graph_2,
                'global_graph3': global_graph_3,
                # 'local_graph': local_graph_,
                'relevant_sentences': relevant_sentences,
                'relevant_sentence_presentation':relevant_sentences_presentation,
                'aspect_sequence_label': aspect_sequence_label,
                'opinion_sequence_label': opinion_sequence_label,
                'sentiment_sequence_label': sentiment_sequence_label,
                'aspect_span_labels': aspect_span_label,
                'opinion_span_labels': opinion_span_label,
                'pair_grid_labels': pair_grid_label,
                'triple_grid_labels': triple_grid_label
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='res14', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'res14': {
                'train': './ASTE-Data-V2/res14/train_triplets.txt',
                'test': './ASTE-Data-V2/res14/test_triplets.txt',
                'dev': './ASTE-Data-V2/res14/dev_triplets.txt'
            },
            'lap14': {
                'train': './ASTE-Data-V2/lap14/train_triplets.txt',
                'test': './ASTE-Data-V2/lap14/test_triplets.txt',
                'dev': './ASTE-Data-V2/lap14/dev_triplets.txt'
            },
            'res15': {
                'train': './ASTE-Data-V2/res15/train_triplets.txt',
                'test': './ASTE-Data-V2/res15/test_triplets.txt',
                'dev': './ASTE-Data-V2/res15/dev_triplets.txt'
            },
            'res16': {
                'train': './ASTE-Data-V2/res16/train_triplets.txt',
                'test': './ASTE-Data-V2/res16/test_triplets.txt',
                'dev': './ASTE-Data-V2/res16/dev_triplets.txt'
            },
            'mams': {
                'train': './ASTE-Data-V2/res16/train_triplets.txt',
                'test': './ASTE-Data-V2/res16/test_triplets.txt',
                'dev': './ASTE-Data-V2/res16/dev_triplets.txt'
            }
        }

        text = ABSADatasetReader.__read_text__([fname[dataset]['train'], fname[dataset]['dev'], fname[dataset]['test']])
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatasetReader.__read_data__(fname=fname[dataset]['train'], domain=dataset, phase='train', tokenizer=tokenizer))
        self.dev_data = ABSADataset(ABSADatasetReader.__read_data__(fname=fname[dataset]['dev'], domain=dataset, phase='dev', tokenizer=tokenizer))
        self.test_data = ABSADataset(ABSADatasetReader.__read_data__(fname=fname[dataset]['test'], domain=dataset, phase='test', tokenizer=tokenizer))

if __name__ == '__main__':
    tokenizer = Tokenizer()
    ABSADatasetReader.__read_data__(fname='./ASTE-Data-V2/res14/train_triplets.txt', domain='res14', phase='train', tokenizer=tokenizer)