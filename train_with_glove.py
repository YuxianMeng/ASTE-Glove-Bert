# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
import pdb
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatasetReader
from models import TS, TS0, TS1, TS2, TS3, TS1_3
from evaluation_glove import get_metric, find_pair, find_term,  compute_sentiment, find_pair_sentiment, find_grid_term
from utils import *
from sklearn.metrics import f1_score, precision_score, accuracy_score
from tqdm import tqdm


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        import datetime as dt
        now_time = dt.datetime.now().strftime('%F %T')
        absa_dataset = ABSADatasetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        # adj, features = load_corpus(dataset_str=opt.dataset)
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.dev_data_loader = BucketIterator(data=absa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False, sort=False)
        
        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self.f_out = open('log/'+ self.opt.dataset + '/' + self.opt.model_name+'_'+self.opt.dataset+'_val'+str(now_time)+'.txt', 'w', encoding='utf-8')
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_aspect_dev_f1, max_opinion_dev_f1 = 0, 0
        max_pair_dev_f1, max_pair_sentiment_dev_f1 = 0, 0
        max_pair_sentiment_dev_f1_macro = 0 

        max_aspect_test_f1, max_opinion_test_f1 = 0, 0
        max_pair_test_f1, max_pair_sentiment_test_f1 = 0, 0
        max_pair_sentiment_test_f1_macro = 0
        max_precision, max_recall = 0, 0

        global_step = 0
        continue_not_increase = 0
        best_results, best_labels = [], []
        for epoch in (range(self.opt.num_epoch)):
            print('>' * 100)
            print('epoch: ', epoch)
            self.f_out.write('>' * 100+'\n')
            self.f_out.write('epoch: {:.4f}\n'.format(epoch))
            loss_g_a, loss_g_o, loss_g_s, loss_g_ag, loss_g_og, loss_g_p, loss_g_ps = 0, 0, 0, 0, 0, 0, 0
            correct_g, predicted_g, relevant_g = 0, 0, 0
            
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]

                targets_aspect_sequence = sample_batched['aspect_sequence_labels'].to(self.opt.device)
                targets_opinion_sequence = sample_batched['opinion_sequence_labels'].to(self.opt.device)
                targets_pair = sample_batched['pair_grid_labels'].to(self.opt.device)
                targets_pair_sentiment = sample_batched['triple_grid_labels'].to(self.opt.device)
                
                mask = sample_batched['mask'].to(self.opt.device)
                aspect_mask = sample_batched['aspect_mask'].to(self.opt.device)
                aspect_mask_ = aspect_mask.reshape(-1).long()
                
                bs, sen_len = mask.size()
                mask_grid = torch.where(mask.unsqueeze(1).repeat(1,sen_len,1) == mask.unsqueeze(-1).repeat(1,1,sen_len),\
                                        torch.ones([bs, sen_len, sen_len]).to(self.opt.device), \
                                        torch.zeros([bs, sen_len, sen_len]).to(self.opt.device))

                outputs_aspect, outputs_opinion, outputs_pair, outputs_pair_sentiment = self.model(inputs, mask)

                outputs_aspect_env = outputs_aspect.argmax(dim=-1)
                outputs_aspect_env = outputs_aspect_env.view(targets_aspect_sequence.shape[0],targets_aspect_sequence.shape[1])
                outputs_aspect_, targets_aspect_ = outputs_aspect.reshape(-1,3), targets_aspect_sequence.reshape(-1).long()

                outputs_opinion_env = outputs_opinion.argmax(dim=-1)
                outputs_opinion_env = outputs_opinion_env.view(targets_opinion_sequence.shape[0],targets_opinion_sequence.shape[1])
                outputs_opinion_, targets_opinion_ = outputs_opinion.reshape(-1,3), targets_opinion_sequence.reshape(-1).long()

                outputs_pair_env = outputs_pair.argmax(dim=-1)
                outputs_pair_env = outputs_pair_env.view(targets_pair.shape[0],targets_pair.shape[1],targets_pair.shape[2])
                outputs_pair_, targets_pair_ = outputs_pair.reshape(-1,3), targets_pair.reshape(-1).long()

                outputs_pair_sentiment_env = outputs_pair_sentiment.argmax(dim=-1)
                outputs_pair_sentiment_env = outputs_pair_sentiment_env.view(targets_pair_sentiment.shape[0],targets_pair_sentiment.shape[1],targets_pair_sentiment.shape[2])
                outputs_pair_sentiment_, targets_pair_sentiment_ = outputs_pair_sentiment.reshape(-1,4), targets_pair_sentiment.reshape(-1).long()
                
                loss_aspect = (criterion(outputs_aspect_, targets_aspect_) * mask).sum() / mask.sum()
                loss_opinion = (criterion(outputs_opinion_, targets_opinion_) * mask).sum() / mask.sum()
                loss_pair = (criterion(outputs_pair_, targets_pair_) * mask_grid).sum() / mask_grid.sum()
                # mask
                loss_pair_sentiment = (criterion(outputs_pair_sentiment_, targets_pair_sentiment_) * mask_grid).sum() / mask_grid.sum()
                # loss_mask = torch.where(targets_pair_>0, True, False)
                # masked_targets_pair_sentiment_ = torch.masked_select(targets_pair_sentiment_, loss_mask, out=None)
                # masked_outputs_pair_sentiment_ = torch.masked_select(outputs_pair_sentiment_, loss_mask.unsqueeze(-1).repeat(1,4), out=None).view(-1, 4)
                # loss_pair_sentiment = criterion(masked_outputs_pair_sentiment_, masked_targets_pair_sentiment_)
                # compute the uncertainty 
                loss_g_a, loss_g_o, loss_g_p, loss_g_ps = \
                    loss_g_a + loss_aspect, loss_g_o + loss_opinion, loss_g_p + loss_pair, loss_g_ps + loss_pair_sentiment 
                loss = loss_aspect + loss_opinion + loss_pair  + loss_pair_sentiment
                # loss = loss_pair  + loss_pair_sentiment
                loss.backward()
                optimizer.step()
            # if epoch == 7:
            #     pdb.set_trace()
            dev_f_aspect, dev_f_opinion, dev_f_pair, dev_f_pair_sentiment, dev_f_pair_sentiment_macro, dev_loss = self._evaluate_acc_f1()
            test_f_aspect, test_f_opinion, test_f_pair, [test_f_pair_sentiment, test_p_pair_sentiment, test_r_pair_sentiment], test_f_pair_sentiment_macro, results, labels, test_loss = self._test_acc_f1()

            print('train loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}'\
                .format(loss_g_a.item(), loss_g_o.item(), loss_g_p.item(), loss_g_ps.item()))
            print('dev loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}'\
                .format(dev_loss[0].item(), dev_loss[1].item(), dev_loss[2].item(), dev_loss[3].item()))
            print('dev: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}'.format(dev_f_aspect, dev_f_opinion, dev_f_pair, dev_f_pair_sentiment, dev_f_pair_sentiment_macro))
            print('test loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}'\
                .format(test_loss[0].item(), test_loss[1].item(), test_loss[2].item(), test_loss[3].item()))
            print('test: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}'.format(test_f_aspect, test_f_opinion, test_f_pair, test_f_pair_sentiment, test_f_pair_sentiment_macro))

            self.f_out.write('train loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}\n'\
                .format(loss_g_a.item(), loss_g_o.item(), loss_g_p.item(), loss_g_ps.item()))
            self.f_out.write('dev loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}\n'\
                .format(dev_loss[0].item(), dev_loss[1].item(), dev_loss[2].item(), dev_loss[3].item()))
            self.f_out.write('dev: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}\n'.format(dev_f_aspect, dev_f_opinion, dev_f_pair, dev_f_pair_sentiment, dev_f_pair_sentiment_macro))
            self.f_out.write('test loss: aspect {:.4f}, opinion {:.4f}, pair {:.4f}, pair_sentiment {:.4f}\n'\
                .format(test_loss[0].item(), test_loss[1].item(), test_loss[2].item(), test_loss[3].item()))
            self.f_out.write('test: f1-aspect: {:.4f}, f1-opinion: {:.4f}, f1-pair: {:.4f}, f1-pair-sentiment: {:.4f}, f1-pair-sentiment-macro: {:.4f}\n'.format(test_f_aspect, test_f_opinion, test_f_pair, test_f_pair_sentiment, test_f_pair_sentiment_macro))
            
            self.f_out.write('test: p-pair-sentiment: {:.4f}, r-pair-sentiment: {:.4f}\n'\
                .format(test_p_pair_sentiment, test_r_pair_sentiment))
            if dev_f_pair_sentiment > max_pair_sentiment_dev_f1:
                max_pair_dev_f1 = dev_f_pair
                max_aspect_dev_f1 = dev_f_aspect
                max_opinion_dev_f1 = dev_f_opinion
                max_pair_sentiment_dev_f1 = dev_f_pair_sentiment
                max_pair_sentiment_dev_f1_macro = dev_f_pair_sentiment_macro
                best_model = self.model

                max_pair_test_f1 = test_f_pair
                max_aspect_test_f1 = test_f_aspect
                max_opinion_test_f1 = test_f_opinion
                max_pair_sentiment_test_f1 = test_f_pair_sentiment
                max_pair_sentiment_test_f1_macro = test_f_pair_sentiment_macro
                best_results = results
                best_labels = labels
                max_precision, max_recall = test_p_pair_sentiment, test_r_pair_sentiment
                self.f_out.write('dev: {:.4f}, test: {:.4f}'.format(max_pair_sentiment_dev_f1, max_pair_sentiment_test_f1))
        return max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1,\
                max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1,\
                max_precision, max_recall,\
                 best_results, best_labels, best_model

    def _evaluate_acc_f1(self):
            # switch model to evaluation mode
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            predicted_p, relevant_p, correct_p = 0, 0, 0
            predicted_ps, relevant_ps, correct_ps = 0, 0, 0
            predicted_a, relevant_a, correct_a = 0, 0, 0
            predicted_o, relevant_o, correct_o = 0, 0, 0

            predicted_ps_macro, relevant_ps_macro, correct_ps_macro = {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}
            dic = {1:'pos', 2:'neg', 3:'neu'}

            loss_g_a, loss_g_o, loss_g_s, loss_g_p, loss_g_ps, loss_g_ag, loss_g_og = 0, 0, 0, 0, 0, 0, 0
            with torch.no_grad():
                for t_batch, t_sample_batched in enumerate(self.dev_data_loader):
                    t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                    t_targets_aspect = t_sample_batched['aspect_sequence_labels'].to(self.opt.device)
                    t_targets_opinion = t_sample_batched['opinion_sequence_labels'].to(self.opt.device)
                    t_targets_sentiment = t_sample_batched['sentiment_sequence_labels'].to(self.opt.device)
                    t_targets_pair = t_sample_batched['pair_grid_labels'].to(self.opt.device)
                    t_targets_pair_sentiment = t_sample_batched['triple_grid_labels'].to(self.opt.device)
                    t_targets_mask = t_sample_batched['mask'].to(self.opt.device)
                    t_aspect_mask = t_sample_batched['aspect_mask'].to(self.opt.device)
                    t_aspect_mask_ = t_aspect_mask.reshape(-1).long()
                    
                    bs, sen_len = t_targets_mask.size()
                    t_targets_mask_grid = torch.where(t_targets_mask.unsqueeze(1).repeat(1,sen_len,1) == t_targets_mask.unsqueeze(-1).repeat(1,1,sen_len),\
                                            torch.ones([bs, sen_len, sen_len]).to(self.opt.device), \
                                            torch.zeros([bs, sen_len, sen_len]).to(self.opt.device))

                    t_outputs_aspect, t_outputs_opinion, t_outputs_pair, t_outputs_pair_sentiment = self.model(t_inputs, t_targets_mask)
                    
                    t_outputs_aspect_env = t_outputs_aspect.argmax(dim=-1).view(t_targets_aspect.shape[0],t_targets_pair.shape[1])
                    t_outputs_opinion_env = t_outputs_opinion.argmax(dim=-1).view(t_targets_opinion.shape[0],t_targets_pair.shape[1])
                    t_outputs_pair_env = t_outputs_pair.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                    t_outputs_pair_sentiment_env = t_outputs_pair_sentiment.argmax(dim=-1).view(t_targets_pair_sentiment.shape[0],t_targets_pair_sentiment.shape[1],t_targets_pair_sentiment.shape[2])
                    # compute loss 
                    outputs_aspect_, targets_aspect_ = t_outputs_aspect.reshape(-1,3), t_targets_aspect.reshape(-1).long()
                    outputs_opinion_, targets_opinion_ = t_outputs_opinion.reshape(-1,3), t_targets_opinion.reshape(-1).long()
                    outputs_pair_, targets_pair_ = t_outputs_pair.reshape(-1,3), t_targets_pair.reshape(-1).long()
                    outputs_pair_sentiment_, targets_pair_sentiment_ = t_outputs_pair_sentiment.reshape(-1,4), t_targets_pair_sentiment.reshape(-1).long()

                    loss_aspect = (criterion(outputs_aspect_, targets_aspect_)*t_targets_mask).sum() / t_targets_mask.sum()
                    loss_opinion = (criterion(outputs_opinion_, targets_opinion_)*t_targets_mask).sum() / t_targets_mask.sum()
                    loss_pair = (criterion(outputs_pair_, targets_pair_)*t_targets_mask_grid).sum() / t_targets_mask_grid.sum()
                    loss_pair_sentiment = (criterion(outputs_pair_sentiment_, targets_pair_sentiment_)*t_targets_mask_grid).sum() / t_targets_mask_grid.sum()
                    
                    loss_g_a, loss_g_o, loss_g_p, loss_g_ps = \
                        loss_g_a + loss_aspect, loss_g_o + loss_opinion, loss_g_p + loss_pair, loss_g_ps + loss_pair_sentiment 
                    # metrics
                    outputs_a = (t_outputs_aspect_env*t_targets_mask).cpu().numpy().tolist()
                    targets_a = t_targets_aspect.cpu().numpy().tolist()

                    outputs_o = (t_outputs_opinion_env*t_targets_mask).cpu().numpy().tolist()
                    targets_o = t_targets_opinion.cpu().numpy().tolist()
                    
                    outputs_p = (t_outputs_pair_env*t_targets_mask_grid).cpu().numpy().tolist()
                    targets_p = t_targets_pair.cpu().numpy().tolist()

                    outputs_ps = (t_outputs_pair_sentiment_env*t_targets_mask_grid).cpu().numpy().tolist()
                    targets_ps = t_targets_pair_sentiment.cpu().numpy().tolist()

                    # f1 for aspect 
                    for out, tar in zip(outputs_a, targets_a):
                        predict_aspect = find_term(out)
                        true_aspect = find_term(tar)
                        predicted_a += len(predict_aspect)
                        relevant_a += len(true_aspect)

                        for aspect in predict_aspect:
                            if aspect in true_aspect:
                                correct_a += 1

                    # f1 for opinion
                    for out, tar in zip(outputs_o, targets_o):
                        predict_opinion = find_term(out)
                        true_opinion = find_term(tar)
                        predicted_o += len(predict_opinion)
                        relevant_o += len(true_opinion)

                        for opinion in predict_opinion:
                            if opinion in true_opinion:
                                correct_o += 1
                    
                    #  f1 for pair
                    for out, tar in zip(outputs_p, targets_p):
                        predict_pairs = find_pair(out)
                        true_pairs = find_pair(tar)
                        predicted_p += len(predict_pairs)
                        relevant_p += len(true_pairs)

                        for pair in predict_pairs:
                            if pair in true_pairs:
                                correct_p += 1
                                
                    # f1 for sentiment pair
                    for out, tar, out_s, tar_s in zip(outputs_p, targets_p, outputs_ps, targets_ps):
                        predict_pairs_sentiment = find_pair_sentiment(out, out_s)
                        true_pairs_sentiment = find_pair_sentiment(tar, tar_s)
                        # micro
                        predicted_ps += len(predict_pairs_sentiment)
                        relevant_ps += len(true_pairs_sentiment)

                        for pair in predict_pairs_sentiment:
                            if pair in true_pairs_sentiment:
                                correct_ps += 1
                        # macro
                        for tri in predict_pairs_sentiment:
                            predicted_ps_macro[dic[tri[2]]]+=1
                        for tri in true_pairs_sentiment:
                            relevant_ps_macro[dic[tri[2]]]+=1
                        for pair in predict_pairs_sentiment:
                            if pair in true_pairs_sentiment:
                                correct_ps_macro[dic[tri[2]]] += 1
                # micro
                p_pair_sentiment = correct_ps / (predicted_ps + 1e-6)
                r_pair_sentiment = correct_ps / (relevant_ps + 1e-6)
                f_pair_sentiment = 2 * p_pair_sentiment * r_pair_sentiment / (p_pair_sentiment + r_pair_sentiment + 1e-6)
                # macro
                p_pair_sentiment_pos, p_pair_sentiment_neg, p_pair_sentiment_neu = \
                    correct_ps_macro['pos'] / (predicted_ps_macro['pos'] + 1e-6), correct_ps_macro['neg'] / (predicted_ps_macro['neg'] + 1e-6), correct_ps_macro['neu'] / (predicted_ps_macro['neu'] + 1e-6)
                r_pair_sentiment_pos, r_pair_sentiment_neg, r_pair_sentiment_neu = \
                    correct_ps_macro['pos'] / (relevant_ps_macro['pos'] + 1e-6), correct_ps_macro['neg'] / (relevant_ps_macro['neg'] + 1e-6), correct_ps_macro['neu'] / (relevant_ps_macro['neu'] + 1e-6)
                f_pair_sentiment_pos, f_pair_sentiment_neg, f_pair_sentiment_neu = \
                    2 * p_pair_sentiment_pos * r_pair_sentiment_pos / (p_pair_sentiment_pos + r_pair_sentiment_pos + 1e-6),\
                        2 * p_pair_sentiment_neg * r_pair_sentiment_neg / (p_pair_sentiment_neg + r_pair_sentiment_neg + 1e-6),\
                            2 * p_pair_sentiment_neu * r_pair_sentiment_neu / (p_pair_sentiment_neu + r_pair_sentiment_neu + 1e-6)
                f_pair_sentiment_macro = (f_pair_sentiment_pos + f_pair_sentiment_neg + f_pair_sentiment_neu) / 3.0

                p_pair = correct_p / (predicted_p + 1e-6)
                r_pair = correct_p / (relevant_p + 1e-6)
                f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-6)

                p_aspect = correct_a / (predicted_a + 1e-6)
                r_aspect = correct_a / (relevant_a + 1e-6)
                f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

                p_opinion = correct_o / (predicted_o + 1e-6)
                r_opinion = correct_o / (relevant_o + 1e-6)
                f_opinion = 2 * p_opinion * r_opinion / (p_opinion + r_opinion + 1e-6)

                return f_aspect, f_opinion, f_pair, f_pair_sentiment, f_pair_sentiment_macro, [loss_g_a, loss_g_o, loss_g_p, loss_g_ps]

    def _test_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        predicted_p, relevant_p, correct_p = 0, 0, 0
        predicted_ps, relevant_ps, correct_ps = 0, 0, 0
        predicted_a, relevant_a, correct_a = 0, 0, 0
        predicted_o, relevant_o, correct_o = 0, 0, 0

        predicted_ps_macro, relevant_ps_macro, correct_ps_macro = {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}, {'pos':0, 'neg':0, 'neu':0}
        dic = {1:'pos', 2:'neg', 3:'neu'}

        loss_g_a, loss_g_o, loss_g_p, loss_g_ps = 0, 0, 0, 0

        aspect_results, opinion_results, sentiment_results, pair_results, pair_sentiment_results = [], [], [], [], [] 
        aspect_labels, opinion_labels, sentiment_labels, pair_labels, pair_sentiment_labels = [], [], [], [], [] 

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets_pair = t_sample_batched['pair_grid_labels'].to(self.opt.device)
                t_targets_pair_sentiment = t_sample_batched['triple_grid_labels'].to(self.opt.device)
                t_targets_aspect = t_sample_batched['aspect_sequence_labels'].to(self.opt.device)
                t_targets_opinion = t_sample_batched['opinion_sequence_labels'].to(self.opt.device)
                t_targets_mask = t_sample_batched['mask'].to(self.opt.device)
                t_aspect_mask = t_sample_batched['aspect_mask'].to(self.opt.device)
                t_aspect_mask_ = t_aspect_mask.reshape(-1).long()

                bs, sen_len = t_targets_mask.size()
                t_targets_mask_grid = torch.where(t_targets_mask.unsqueeze(1).repeat(1,sen_len,1) == t_targets_mask.unsqueeze(-1).repeat(1,1,sen_len),\
                                        torch.ones([bs, sen_len, sen_len]).to(self.opt.device), \
                                        torch.zeros([bs, sen_len, sen_len]).to(self.opt.device))

                t_outputs_aspect, t_outputs_opinion, t_outputs_pair, t_outputs_pair_sentiment = self.model(t_inputs, t_targets_mask)
                
                t_outputs_aspect_env = t_outputs_aspect.argmax(dim=-1).view(t_targets_aspect.shape[0],t_targets_pair.shape[1])
                t_outputs_opinion_env = t_outputs_opinion.argmax(dim=-1).view(t_targets_opinion.shape[0],t_targets_pair.shape[1])
                t_outputs_pair_env = t_outputs_pair.argmax(dim=-1).view(t_targets_pair.shape[0],t_targets_pair.shape[1],t_targets_pair.shape[2])
                t_outputs_pair_sentiment_env = t_outputs_pair_sentiment.argmax(dim=-1).view(t_targets_pair_sentiment.shape[0],t_targets_pair_sentiment.shape[1],t_targets_pair_sentiment.shape[2])
                # compute loss 
                outputs_aspect_, targets_aspect_ = t_outputs_aspect.reshape(-1,3), t_targets_aspect.reshape(-1).long()
                outputs_opinion_, targets_opinion_ = t_outputs_opinion.reshape(-1,3), t_targets_opinion.reshape(-1).long()
                outputs_pair_, targets_pair_ = t_outputs_pair.reshape(-1,3), t_targets_pair.reshape(-1).long()
                outputs_pair_sentiment_, targets_pair_sentiment_ = t_outputs_pair_sentiment.reshape(-1,4), t_targets_pair_sentiment.reshape(-1).long()

                loss_aspect = (criterion(outputs_aspect_, targets_aspect_)*t_targets_mask).sum() / t_targets_mask.sum()
                loss_opinion = (criterion(outputs_opinion_, targets_opinion_)*t_targets_mask).sum() / t_targets_mask.sum()
                loss_pair = (criterion(outputs_pair_, targets_pair_)*t_targets_mask_grid).sum() / t_targets_mask_grid.sum()
                loss_pair_sentiment = (criterion(outputs_pair_sentiment_, targets_pair_sentiment_)*t_targets_mask_grid).sum() / t_targets_mask_grid.sum()

                loss_g_a, loss_g_o, loss_g_p, loss_g_ps = \
                    loss_g_a + loss_aspect, loss_g_o + loss_opinion, loss_g_p + loss_pair, loss_g_ps + loss_pair_sentiment 
                # metrics
                outputs_a = (t_outputs_aspect_env*t_targets_mask).cpu().numpy().tolist()
                targets_a = t_targets_aspect.cpu().numpy().tolist()

                outputs_o = (t_outputs_opinion_env*t_targets_mask).cpu().numpy().tolist()
                targets_o = t_targets_opinion.cpu().numpy().tolist()
                
                outputs_p = (t_outputs_pair_env*t_targets_mask_grid).cpu().numpy().tolist()
                targets_p = t_targets_pair.cpu().numpy().tolist()

                outputs_ps = (t_outputs_pair_sentiment_env*t_targets_mask_grid).cpu().numpy().tolist()
                targets_ps = t_targets_pair_sentiment.cpu().numpy().tolist()


                # f1 for aspect 
                for out, tar in zip(outputs_a, targets_a):
                    predict_aspect = find_term(out)
                    true_aspect = find_term(tar)
                    predicted_a += len(predict_aspect)
                    relevant_a += len(true_aspect)

                    for aspect in predict_aspect:
                        if aspect in true_aspect:
                            correct_a += 1

                # f1 for opinion
                for out, tar in zip(outputs_o, targets_o):
                    predict_opinion = find_term(out)
                    true_opinion = find_term(tar)
                    predicted_o += len(predict_opinion)
                    relevant_o += len(true_opinion)

                    for opinion in predict_opinion:
                        if opinion in true_opinion:
                            correct_o += 1
                
                #  f1 for pair
                for out, tar in zip(outputs_p, targets_p):
                    predict_pairs = find_pair(out)
                    true_pairs = find_pair(tar)
                    predicted_p += len(predict_pairs)
                    relevant_p += len(true_pairs)

                    for pair in predict_pairs:
                        if pair in true_pairs:
                            correct_p += 1

                # f1 for sentiment pair
                for out, tar, out_s, tar_s in zip(outputs_p, targets_p, outputs_ps, targets_ps):
                    predict_pairs_sentiment = find_pair_sentiment(out, out_s)
                    true_pairs_sentiment = find_pair_sentiment(tar, tar_s)
                    # micro
                    predicted_ps += len(predict_pairs_sentiment)
                    relevant_ps += len(true_pairs_sentiment)

                    for pair in predict_pairs_sentiment:
                        if pair in true_pairs_sentiment:
                            correct_ps += 1
                    # macro
                        for tri in predict_pairs_sentiment:
                            predicted_ps_macro[dic[tri[2]]]+=1
                        for tri in true_pairs_sentiment:
                            relevant_ps_macro[dic[tri[2]]]+=1
                        for pair in predict_pairs_sentiment:
                            if pair in true_pairs_sentiment:
                                correct_ps_macro[dic[tri[2]]] += 1

                # save results and labels
                aspect_results.append(t_outputs_aspect.view(t_targets_pair.shape[0], -1, 3).cpu().numpy().tolist())
                opinion_results.append(t_outputs_opinion.view(t_targets_pair.shape[0], -1, 3).cpu().numpy().tolist())
                pair_results.append(t_outputs_pair.view(t_targets_pair.shape[0], t_targets_aspect.shape[-1], t_targets_aspect.shape[-1], 3).cpu().numpy().tolist())
                pair_sentiment_results.append(t_outputs_pair_sentiment.view(t_targets_pair.shape[0], t_targets_aspect.shape[-1], t_targets_aspect.shape[-1], 4).cpu().numpy().tolist())

                aspect_labels.append(t_targets_aspect.cpu().numpy().tolist())
                opinion_labels.append(t_targets_opinion.cpu().numpy().tolist())
                pair_labels.append(t_targets_pair.cpu().numpy().tolist())
                pair_sentiment_labels.append(t_targets_pair_sentiment.cpu().numpy().tolist())
            # micro
            p_pair_sentiment = correct_ps / (predicted_ps + 1e-6)
            r_pair_sentiment = correct_ps / (relevant_ps + 1e-6)
            f_pair_sentiment = 2 * p_pair_sentiment * r_pair_sentiment / (p_pair_sentiment + r_pair_sentiment + 1e-6)
            # macro
            p_pair_sentiment_pos, p_pair_sentiment_neg, p_pair_sentiment_neu = \
                correct_ps_macro['pos'] / (predicted_ps_macro['pos'] + 1e-6), correct_ps_macro['neg'] / (predicted_ps_macro['neg'] + 1e-6), correct_ps_macro['neu'] / (predicted_ps_macro['neu'] + 1e-6)
            r_pair_sentiment_pos, r_pair_sentiment_neg, r_pair_sentiment_neu = \
                correct_ps_macro['pos'] / (relevant_ps_macro['pos'] + 1e-6), correct_ps_macro['neg'] / (relevant_ps_macro['neg'] + 1e-6), correct_ps_macro['neu'] / (relevant_ps_macro['neu'] + 1e-6)
            f_pair_sentiment_pos, f_pair_sentiment_neg, f_pair_sentiment_neu = \
                2 * p_pair_sentiment_pos * r_pair_sentiment_pos / (p_pair_sentiment_pos + r_pair_sentiment_pos + 1e-6),\
                    2 * p_pair_sentiment_neg * r_pair_sentiment_neg / (p_pair_sentiment_neg + r_pair_sentiment_neg + 1e-6),\
                        2 * p_pair_sentiment_neu * r_pair_sentiment_neu / (p_pair_sentiment_neu + r_pair_sentiment_neu + 1e-6)
            f_pair_sentiment_macro = (f_pair_sentiment_pos + f_pair_sentiment_neg + f_pair_sentiment_neu) / 3.0
            
            
            p_pair = correct_p / (predicted_p + 1e-6)
            r_pair = correct_p / (relevant_p + 1e-6)
            f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-6)

            p_aspect = correct_a / (predicted_a + 1e-6)
            r_aspect = correct_a / (relevant_a + 1e-6)
            f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

            p_opinion = correct_o / (predicted_o + 1e-6)
            r_opinion = correct_o / (relevant_o + 1e-6)
            f_opinion = 2 * p_opinion * r_opinion / (p_opinion + r_opinion + 1e-6)

            results = [aspect_results, opinion_results, sentiment_results, pair_results, pair_sentiment_results]
            labels = [aspect_labels, opinion_labels, sentiment_labels, pair_labels, pair_sentiment_labels]

            return f_aspect, f_opinion, f_pair, [f_pair_sentiment, p_pair_sentiment, r_pair_sentiment], f_pair_sentiment_macro, results, labels, [loss_g_a, loss_g_o, loss_g_p, loss_g_ps]

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        if not os.path.exists('log/'):
            os.mkdir('log/')

        import datetime as dt
        now_time = dt.datetime.now().strftime('%F %T')

        # f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val'+str(now_time)+'.txt', 'w', encoding='utf-8')
        
        # print args
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.f_out.write('n_trainable_params: {0}, n_nontrainable_params: {1}\n'.format(n_trainable_params, n_nontrainable_params)+'\n')
        self.f_out.write('> training arguments:\n')
        
        for arg in vars(self.opt):
            self.f_out.write('>>> {0}: {1}'.format(arg, getattr(self.opt, arg))+'\n')
        max_aspect_test_f1_avg = 0
        max_opinion_test_f1_avg = 0
        max_sentiment_test_f1_avg = 0
        max_absa_test_f1_avg = 0
        max_pair_test_f1_avg = 0
        max_pair_sentiment_test_f1_avg = 0
        max_precision_avg, max_recall_avg = 0, 0
        for i in range(self.opt.repeats):
            repeats = self.opt.repeats
            print('repeat: ', (i+1))
            self.f_out.write('repeat: '+str(i+1)+'\n')
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            # _params = self.model.parameters()
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            # max_pair_dev_f1, max_aspect_dev_f1, max_opinion_dev_f1, max_pair_test_f1, max_aspect_test_f1, max_opinion_test_f1 = self._train(criterion, optimizer)
            max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1,\
             max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1, max_precision, max_recall, best_results, best_labels, best_model = self._train(criterion, optimizer)

            if self.opt.save_model == 1:
                torch.save(best_model.bert_model, './save_bert_model/' + self.opt.model_name + '_' + self.opt.dataset + '.pkl')

            if self.opt.write_results == 1:
                results_a, results_o, results_s, results_p, results_ps = best_results
                labels_a, labels_o, labels_s, labels_p, labels_ps = best_labels
                # write results
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_a.npy', results_a)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_o.npy', results_o)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_s.npy', results_s)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_p.npy', results_p)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'results_ps.npy', results_ps)
                # write labels
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_a.npy', labels_a)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_o.npy', labels_o)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_s.npy', labels_s)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_p.npy', labels_p)
                np.save('./write_results/'+self.opt.dataset+'/'+self.opt.dataset+'labels_ps.npy', labels_ps)
            print('max_aspect_dev_f1: {:.4f}, max_opinion_dev_f1: {:.4f}, max_pair_dev_f1: {:.4f}, max_pair_sentiment_dev_f1: {:.4f}'.format(max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1))
            print('max_aspect_test_f1: {:.4f}, max_opinion_test_f1: {:.4f}, max_pair_test_f1: {:.4f}, max_pair_sentiment_test_f1: {:.4f}'.format(max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1))
            self.f_out.write('max_aspect_dev_f1: {:.4f}, max_opinion_dev_f1: {:.4f}, max_pair_dev_f1: {:.4f}, max_pair_sentiment_dev_f1: {:.4f}\n'\
                .format(max_aspect_dev_f1, max_opinion_dev_f1, max_pair_dev_f1, max_pair_sentiment_dev_f1)+'\n')
            self.f_out.write('max_aspect_test_f1: {:.4f}, max_opinion_test_f1: {:.4f}, max_pair_test_f1: {:.4f}, max_pair_sentiment_test_f1: {:.4f}\n'\
                .format(max_aspect_test_f1, max_opinion_test_f1, max_pair_test_f1, max_pair_sentiment_test_f1)+'\n')
            self.f_out.write('max_test_precision: {:.4f}, max_test_recall: {:.4f}\n'\
                .format(max_precision, max_recall)+'\n')
            max_aspect_test_f1_avg += max_aspect_test_f1
            max_opinion_test_f1_avg += max_opinion_test_f1
            max_pair_test_f1_avg += max_pair_test_f1
            max_pair_sentiment_test_f1_avg += max_pair_sentiment_test_f1
            max_precision_avg += max_precision
            max_recall_avg += max_recall
            print('#' * 100)

        print("max_aspect_test_f1_avg:", max_aspect_test_f1_avg / repeats)
        print("max_opinion_test_f1_avg:", max_opinion_test_f1_avg / repeats)
        print("max_pair_test_f1_avg:", max_pair_test_f1_avg / repeats)
        print("max_pair_sentiment_test_f1_avg:", max_pair_sentiment_test_f1_avg / repeats)

        self.f_out.write("max_aspect_test_f1_avg:"+ str(max_aspect_test_f1_avg / repeats) + '\n')
        self.f_out.write("max_opinion_test_f1_avg:"+ str(max_opinion_test_f1_avg / repeats) + '\n')
        self.f_out.write("max_pair_test_f1_avg:" + str(max_pair_test_f1_avg / repeats) + '\n')
        self.f_out.write("max_pair_sentiment_test_f1_avg:" + str(max_pair_sentiment_test_f1_avg / repeats) + '\n')
        self.f_out.write("max_precision_avg:" + str(max_precision_avg / repeats) + '\n')
        self.f_out.write("max_recall_avg:" + str(max_recall_avg / repeats) + '\n')

        self.f_out.close()

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ts', type=str)
    parser.add_argument('--dataset', default='lap14', type=str, help='res14, lap14, res15')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--repeats', default=3, type=int)
    parser.add_argument('--use_graph0', default=1, type=int)
    parser.add_argument('--use_graph1', default=0, type=int)
    parser.add_argument('--use_graph2', default=0, type=int)
    parser.add_argument('--use_graph3', default=0, type=int)
    parser.add_argument('--write_results', default=0, type=int)
    parser.add_argument('--save_model', default=0, type=int)
    parser.add_argument('--emb_for_ao', default='private_single', type=str, help='private_single, private_multi, shared_multi' )
    parser.add_argument('--emb_for_ps', default='private_single', type=str, help='private_single, private_multi, shared_multi' )
    parser.add_argument('--use_aspect_opinion_sequence_mask', default=0, type=int, help='1: use the predicted aspect_sequence_label and opinion_sequence_label to construct a grid mask for pair prediction.' )

    parser.add_argument('--gcn_layers_in_graph0', default=1, type=int, help='1 or 2' )
    opt = parser.parse_args()

    model_classes = {
        'ts': TS,
        'ts0': TS0,
        'ts1': TS1,
        'ts2': TS2,
        'ts3': TS3,
        'ts1_3': TS1_3
    }
    input_colses = {
        'ts': ['text_indices', 'mask', 'aspect_sequence_labels','opinion_sequence_labels','sentiment_sequence_labels'],\
        'ts0': ['text_indices', 'mask', 'global_graph0', 'relevant_sentences', 'relevant_sentences_presentation', \
            'pair_grid_labels', 'triple_grid_labels', 'aspect_sequence_labels','opinion_sequence_labels','sentiment_sequence_labels'],\
        'ts1': ['text_indices', 'mask',  'global_graph0', 'global_graph1', 'relevant_sentences', 'relevant_sentences_presentation', \
            'pair_grid_labels', 'triple_grid_labels',  'aspect_sequence_labels','opinion_sequence_labels','sentiment_sequence_labels'],\
        'ts2': ['text_indices', 'mask', 'global_graph0', 'global_graph1', 'global_graph2', 'relevant_sentences', 'relevant_sentences_presentation', \
            'pair_grid_labels', 'triple_grid_labels',  'aspect_sequence_labels','opinion_sequence_labels','sentiment_sequence_labels'],\
        'ts3': ['text_indices', 'mask', 'global_graph0', 'global_graph1', 'global_graph3', 'relevant_sentences', 'relevant_sentences_presentation', \
            'pair_grid_labels', 'triple_grid_labels', 'aspect_sequence_labels','opinion_sequence_labels','sentiment_sequence_labels'],\
        'ts1_3': ['text_indices', 'mask', 'global_graph0', 'global_graph1', 'global_graph3', 'relevant_sentences', 'relevant_sentences_presentation', \
            'pair_grid_labels', 'triple_grid_labels', 'aspect_sequence_labels','opinion_sequence_labels','sentiment_sequence_labels']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
