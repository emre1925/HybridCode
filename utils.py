import torch
import torch.nn as nn
import math, pdb
from torch.autograd import Variable
import argparse
import numpy as np


# fixed PE
class PositionalEncoder_fixed(nn.Module):
    def __init__(self, lenWord=32, max_seq_len=200, dropout=0.0):
        super().__init__()
        self.lenWord = lenWord
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, lenWord)
        for pos in range(max_seq_len):
            for i in range(0, lenWord, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / lenWord)))
                if lenWord != 1:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / lenWord)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.lenWord)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)


# learnable PE
class PositionalEncoder(nn.Module):
    def __init__(self, SeqLen=51, lenWord=64):
        super().__init__()
        self.lenWord = lenWord
        self.pe = torch.nn.Parameter(torch.Tensor(51, lenWord), requires_grad=True)
        self.pe.data.uniform_(0.0, 1.0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]




class Power_reallocate(torch.nn.Module):
    def __init__(self, args):
        super(Power_reallocate, self).__init__()
        self.args = args
        self.weight1 = torch.nn.Parameter(torch.Tensor(args.numb_block, 1), requires_grad=True)
        self.weight1.data.uniform_(1.0, 1.0)
        if args.seq_reloc == 1:
           self.weight2 = torch.nn.Parameter(torch.Tensor(args.parity_pb+args.block_size, 1), requires_grad=True)
           self.weight2.data.uniform_(1.0, 1.0)
           
            
    def forward(self, inputs, seq_order):
       # phase-level power allocation
        self.wt1 = torch.sqrt(self.weight1 ** 2 * (self.args.numb_block / torch.sum(self.weight1 ** 2)))
        if self.args.seq_reloc == 1:
            self.wt2 = torch.sqrt(self.weight2 ** 2 * ((self.args.parity_pb + self.args.block_size) / torch.sum(self.weight2 ** 2)))
        inputs1 = inputs * self.wt1 # block_wise scaling
        if self.args.seq_reloc == 1:
            inputs1 = inputs1 * self.wt2[seq_order] # sequence_wise scaling

        return inputs1
    
    

# BP Decoding algorithm (Switch from TF to Pytorch ==> In progress)

def compute_vc(cv, iteration, soft_input):
    weighted_soft_input = soft_input

    edges = []
    for i in range(0, n):
        for j in range(0, var_degrees[i]):
            edges.append(i)
    reordered_soft_input = tf.gather(weighted_soft_input, edges)

    vc = []
    edge_order = []
    for i in range(0, n): # for each variable node v
        for j in range(0, var_degrees[i]):
            # edge = d[i][j]
            edge_order.append(d[i][j])
            extrinsic_edges = []
            for jj in range(0, var_degrees[i]):
                if jj != j: # extrinsic information only
                    extrinsic_edges.append(d[i][jj])
            # if the list of edges is not empty, add them up
            if extrinsic_edges:
                temp = tf.gather(cv,extrinsic_edges)
                temp = tf.reduce_sum(temp,0)
                #print(temp.shape)

                #temp = temp * tf.tile(tf.reshape(decoder.W_check[iteration,i],[-1,1]),[1,batch_size])

            else:
                temp = tf.zeros([batch_size])
            if SUM_PRODUCT: temp = tf.cast(temp, tf.float32)#tf.cast(temp, tf.float64)
            vc.append(temp)


    vc = tf.stack(vc)
    new_order = np.zeros(num_edges).astype(np.int)
    new_order[edge_order] = np.array(range(0,num_edges)).astype(np.int)
    vc = tf.gather(vc,new_order)

    if W_vc_indicator:
        W_vc = tf.tile(tf.reshape(decoder.W_vc,[-1,1]),[1,batch_size])
        vc = (W_vc * vc) + reordered_soft_input
    else:
        vc = vc + reordered_soft_input

    return vc, cv

# compute messages from check nodes to variable nodes
def compute_cv(vc, cv, iteration):
    cv_list = []
    prod_list = []
    min_list = []
    cv_tmp = cv[:]

    if SUM_PRODUCT:
        vc = tf.clip_by_value(vc, -10, 10)
        tanh_vc = tf.tanh(vc / 2.0)
    edge_order = []
    for i in range(0, m): # for each check node c
        for j in range(0, chk_degrees[i]):
            # edge = u[i][j]
            edge_order.append(u[i][j])
            extrinsic_edges = []
            for jj in range(0, chk_degrees[i]):
                if jj != j:
                    #print(jj)
                    extrinsic_edges.append(u[i][jj])
            if SUM_PRODUCT:
                temp = tf.gather(tanh_vc,extrinsic_edges)
                temp = tf.reduce_prod(temp,0)
                temp = tf.log((1+temp)/(1-temp))
                cv_list.append(temp)
            if MIN_SUM:
                temp = tf.gather(vc,extrinsic_edges)
                temp1 = tf.reduce_prod(tf.sign(temp),0)
                temp2 = tf.reduce_min(tf.abs(temp),0)
                prod_list.append(temp1)
                min_list.append(temp2)


    if SUM_PRODUCT:
        cv = tf.stack(cv_list)

    if MIN_SUM:
        prods = tf.stack(prod_list)
        mins = tf.stack(min_list)
        if decoder.decoder_type == "RNOMS":
            # offsets = tf.nn.softplus(decoder.B_cv)
            # mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
            mins = tf.nn.relu(mins - decoder.B_cv)
        elif decoder.decoder_type == "FNOMS":
            offsets = tf.nn.softplus(decoder.B_cv[iteration])
            mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
        cv = prods * mins

    new_order = np.zeros(num_edges).astype(np.int)
    new_order[edge_order] = np.array(range(0,num_edges)).astype(np.int)
    cv = tf.gather(cv,new_order)

    if W_cv_indicator:
        if decoder.decoder_type == "RNSPA" or decoder.decoder_type == "RNNMS":
            W_cv = tf.tile(tf.reshape(decoder.W_cv,[-1,1]),[1,batch_size])
            cv = cv * W_cv
        elif decoder.decoder_type == "FNSPA" or decoder.decoder_type == "FNNMS":
            W_cv = tf.tile(tf.reshape(decoder.W_cv[iteration],[-1,1]),[1,batch_size])
            cv = cv * W_cv
        elif decoder.decoder_type == "RNN-SS":
            W_cv = tf.tile(tf.reshape(decoder.W_cv,[-1,1]),[num_edges,batch_size])
            if SNR_adaptation:
                predictions_cv = PAN(decoder.snr, decoder.weights_CV, decoder.biases_CV)
                cv = cv * W_cv * predictions_cv
            else:
                cv = cv * W_cv       
    return cv

# combine messages to get posterior LLRs
def marginalize(soft_input, iteration, cv):
    weighted_soft_input = soft_input

    soft_output = []
    for i in range(0,n):
        edges = []
        for e in range(0,var_degrees[i]):
            edges.append(d[i][e])

        temp = tf.gather(cv,edges)
        temp = tf.reduce_sum(temp,0)
        soft_output.append(temp)

    soft_output = tf.stack(soft_output)

    soft_output = weighted_soft_input + soft_output
    return soft_output


def belief_propagation_iteration(soft_input, soft_output, iteration, cv, m_t, loss, labels, syndrome_wt, indicator, soft_term):
        
    # compute vc
    vc, cv = compute_vc(cv,iteration,soft_input)

    # filter vc
    if decoder.relaxed:
        if SNR_adaptation:
            predictions_relaxation = PAN(decoder.snr, decoder.weights_Relaxation, decoder.biases_Relaxation)
            m_t = (R * predictions_relaxation *m_t) + ((1-(R * predictions_relaxation)) * vc)
        else:
            m_t = R * m_t + (1-R) * vc
        vc_prime = m_t
    else:
        vc_prime = vc

    # compute cv
    cv = compute_cv(vc_prime, cv, iteration)

    # get output for this iteration
    soft_output = marginalize(soft_input, iteration, cv)
   
    # L = 0.5
    print("L = " + str(L))
    CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) / num_iterations
    loss = loss + CE_loss

    iteration += 1

    return soft_input, soft_output, iteration, cv, m_t, loss, labels, syndrome_wt, indicator, soft_term


