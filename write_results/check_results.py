import numpy as np
import pdb
import sys 
import torch
sys.path.append("..") 
from evaluation_bert import find_pair, find_term, find_pair_sentiment

domain = 'res14'

a_l = np.load(domain + '/' + domain + 'labels_a.npy', allow_pickle=True)
a_r = np.load(domain + '/' + domain + 'results_a.npy', allow_pickle=True)
o_l = np.load(domain + '/' + domain + 'labels_o.npy', allow_pickle=True)
o_r = np.load(domain + '/' + domain + 'results_o.npy', allow_pickle=True)
p_l = np.load(domain + '/' + domain + 'labels_p.npy', allow_pickle=True)
p_r = np.load(domain + '/' + domain + 'results_p.npy', allow_pickle=True)
ps_l = np.load(domain + '/' + domain + 'labels_ps.npy', allow_pickle=True)
ps_r = np.load(domain + '/' + domain + 'results_ps.npy', allow_pickle=True)

a_l = [l for sub_list in a_l for l in sub_list]
a_r = [l for sub_list in a_r for l in sub_list]
o_l = [l for sub_list in o_l for l in sub_list]
o_r = [l for sub_list in o_r for l in sub_list]
p_l = [l for sub_list in p_l for l in sub_list]
p_r = [l for sub_list in p_r for l in sub_list]
ps_l = [l for sub_list in ps_l for l in sub_list]
ps_r = [l for sub_list in ps_r for l in sub_list]

results = open(domain + '_results.txt', 'a')

for i in range(len(a_l)):
    aspect_l, aspect_r = find_term(a_l[i]), find_term(torch.argmax(torch.tensor(a_r[i]), -1))
    opinion_l, opinion_r = find_term(o_l[i]), find_term(torch.argmax(torch.tensor(o_r[i]), -1))
    pair_l, pair_r = find_pair(p_l[i]), find_pair(torch.argmax(torch.tensor(p_r[i]),-1))
    pair_sentiment_l, pair_sentiment_r = find_pair_sentiment(p_l[i], ps_l[i]), find_pair_sentiment(torch.argmax(torch.tensor(p_r[i]),-1), torch.argmax(torch.tensor(ps_r[i]),-1))
    for tri in pair_sentiment_r:
        tri[-1] = tri[-1].item()
    if pair_sentiment_l != pair_sentiment_r or pair_l != pair_r:
        results.write(str(i)+'\n')
        results.write('True:' + str(pair_sentiment_l) + '\n')
        results.write('Results:' + str(pair_sentiment_r) + '\n')
    # pdb.set_trace()
results.close()