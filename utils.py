import torch
import torch.nn as nn
import math, pdb
from torch.autograd import Variable
import argparse
import numpy as np
import commpy.channelcoding.ldpc as ldpc


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
    
    
# Extract the code parameters from the header file   
def get_ldpc_code_params(ldpc_design_filename, compute_matrix=False):
    with open(ldpc_design_filename) as ldpc_design_file:

        [n_vnodes, n_cnodes] = [int(x) for x in ldpc_design_file.readline().split(' ')]
        [max_vnode_deg, max_cnode_deg] = [int(x) for x in ldpc_design_file.readline().split(' ')]
        vnode_deg_list = np.array([int(x) for x in ldpc_design_file.readline().split(' ')[:-1]], np.int32)
        cnode_deg_list = np.array([int(x) for x in ldpc_design_file.readline().split(' ')[:-1]], np.int32)

        cnode_adj_list = -np.ones([n_cnodes, max_cnode_deg], int)
        vnode_adj_list = -np.ones([n_vnodes, max_vnode_deg], int)

        for vnode_idx in range(n_vnodes):
            vnode_adj_list[vnode_idx, 0:vnode_deg_list[vnode_idx]] = \
                np.array([int(x)-1 for x in ldpc_design_file.readline().split(' ')])

        for cnode_idx in range(n_cnodes):
            cnode_adj_list[cnode_idx, 0:cnode_deg_list[cnode_idx]] = \
                np.array([int(x)-1 for x in ldpc_design_file.readline().split(' ')])

    cnode_vnode_map = -np.ones([n_cnodes, max_cnode_deg], int)
    vnode_cnode_map = -np.ones([n_vnodes, max_vnode_deg], int)

    for cnode in range(n_cnodes):
        for i, vnode in enumerate(cnode_adj_list[cnode, 0:cnode_deg_list[cnode]]):
            cnode_vnode_map[cnode, i] = np.where(vnode_adj_list[vnode, :] == cnode)[0]

    for vnode in range(n_vnodes):
        for i, cnode in enumerate(vnode_adj_list[vnode, 0:vnode_deg_list[vnode]]):
            vnode_cnode_map[vnode, i] = np.where(cnode_adj_list[cnode, :] == vnode)[0]

    cnode_adj_list_1d = cnode_adj_list.flatten().astype(np.int32)
    vnode_adj_list_1d = vnode_adj_list.flatten().astype(np.int32)
    cnode_vnode_map_1d = cnode_vnode_map.flatten().astype(np.int32)
    vnode_cnode_map_1d = vnode_cnode_map.flatten().astype(np.int32)

    ldpc_code_params = {}

    ldpc_code_params['n_vnodes'] = n_vnodes
    ldpc_code_params['n_cnodes'] = n_cnodes
    ldpc_code_params['max_cnode_deg'] = max_cnode_deg
    ldpc_code_params['max_vnode_deg'] = max_vnode_deg
    ldpc_code_params['cnode_adj_list'] = cnode_adj_list_1d
    ldpc_code_params['cnode_vnode_map'] = cnode_vnode_map_1d
    ldpc_code_params['vnode_adj_list'] = vnode_adj_list_1d
    ldpc_code_params['vnode_cnode_map'] = vnode_cnode_map_1d
    ldpc_code_params['cnode_deg_list'] = cnode_deg_list
    ldpc_code_params['vnode_deg_list'] = vnode_deg_list

    if compute_matrix:
        ldpc.build_matrix(ldpc_code_params)

    return ldpc_code_params


# BP-SP Decoding algorithm

def ldpc_bp_decode(llr_vec, ldpc_code_params, H, decoder_algorithm, n_iters):
    _llr_max = 500
    B, N, L = llr_vec.size()
    out_llrs = llr_vec.clone()
    out_word = torch.signbit(llr_vec)
    n_c = ldpc_code_params['n_cnodes']
    n_v = ldpc_code_params['n_vnodes']
    recover_indices = torch.arange(0, n_c*N, n_c).to(llr_vec.device)

    llr_vec = llr_vec.clamp(-_llr_max, _llr_max)  # clip LLRs
    llr_vec = llr_vec.repeat_interleave(n_c, dim=1)

    # Initialization
    dec_word = torch.signbit(llr_vec)
    msg_llrs = llr_vec.clone()

    parity_check_matrix = H.to(torch.float).unsqueeze(0).repeat_interleave(B, dim=0)
    parity_check_matrix = parity_check_matrix.repeat(1, N, 1).to_sparse()

    message_matrix = sparse_dense_mul(parity_check_matrix, llr_vec)

    for iter_cnt in range(n_iters):
        iterations = iter_cnt + 1
        term_check = sparse_dense_mul(parity_check_matrix, dec_word)
        term_check = torch.sparse.sum(term_check, dim=2)
        term_check = term_check.to_dense().view(B, N, n_c) % 2
        term_check = torch.chunk(term_check, chunks=B, dim=0)

        curr_llr = torch.index_select(msg_llrs, 1, recover_indices)
        curr_word = torch.index_select(dec_word, 1, recover_indices)
        nan_mask = torch.isnan(curr_llr)
        out_llrs[~nan_mask] = curr_llr[~nan_mask]
        out_word[~nan_mask] = curr_word[~nan_mask]

        if decoder_algorithm == 'SPA':
            message_matrix *= .5
            message_matrix = torch.sparse_coo_tensor(message_matrix._indices(),
                                                     torch.tanh(message_matrix._values()),
                                                     message_matrix.size())

            log2_msg_matrix = torch.sparse_coo_tensor(message_matrix._indices(),
                                                      torch.log2(message_matrix._values().to(torch.complex128)),
                                                      message_matrix.size())

            msg_products_real = torch.sparse_coo_tensor(log2_msg_matrix._indices(),
                                                        log2_msg_matrix._values().real,
                                                        log2_msg_matrix.size())
            msg_products_real = torch.sparse.sum(msg_products_real, dim=2)
            msg_products_imag = torch.sparse_coo_tensor(log2_msg_matrix._indices(),
                                                        log2_msg_matrix._values().imag,
                                                        log2_msg_matrix.size())
            msg_products_imag = torch.sparse.sum(msg_products_imag, dim=2)
            msg_products = torch.sparse_coo_tensor(msg_products_real._indices(),
                                                   torch.view_as_complex(
                                                       torch.stack((msg_products_real._values(),
                                                                    msg_products_imag._values()),
                                                                   dim=1)),
                                                   msg_products_real.size())
            msg_products = torch.sparse_coo_tensor(msg_products._indices(),
                                                   (2 ** msg_products._values()).real,
                                                   msg_products.size())

            message_matrix = torch.sparse_coo_tensor(message_matrix._indices(),
                                                     (1 / message_matrix._values()),
                                                     message_matrix.size())

            message_matrix = sparse_dense_mul(message_matrix,
                                              msg_products.to_dense().unsqueeze(2).repeat_interleave(n_v, dim=2))

            message_matrix = torch.sparse_coo_tensor(message_matrix._indices(),
                                                     message_matrix._values().clamp(-1, 1),
                                                     message_matrix.size())
            message_matrix = torch.sparse_coo_tensor(message_matrix._indices(),
                                                     torch.arctan(message_matrix._values()),
                                                     message_matrix.size())

            message_matrix *= 2
            message_matrix = torch.sparse_coo_tensor(message_matrix._indices(),
                                                     message_matrix._values().clamp(-_llr_max, _llr_max),
                                                     message_matrix.size())
        else:
            raise NotImplementedError('Only SPA is implemented for now')

        msg_sum = message_matrix.to_dense()
        msg_sum = msg_sum.view(B, N, -1, n_v)
        msg_sum = msg_sum.sum(2)
        msg_sum = msg_sum.repeat_interleave(n_c, dim=1)

        message_matrix *= -1
        message_matrix = torch.sparse_coo_tensor(message_matrix._indices(),
                                                 (message_matrix._values()
                                                  + sparse_dense_mul(parity_check_matrix, (msg_sum + llr_vec))._values()),
                                                 message_matrix.size())

        msg_llrs = (msg_sum + llr_vec).to(torch.float)
        dec_word = torch.signbit(msg_llrs)

    return out_word.to(torch.float), out_llrs.to(torch.float32), iterations



class LDPC:
    def __init__(self, header_fn, decode_iters, device):
        self.ldpc_design = self.load_code_from_alist(header_fn)
        self.G = torch.from_numpy(self.ldpc_design['generator_matrix'].A).to(device).transpose(1, 0)
        self.H = torch.from_numpy(self.ldpc_design['parity_check_matrix'].A).to(device)
        self.k = self.G.shape[0]
        self.n = self.H.shape[1]
        self.decode_algorithm = 'SPA'  # TODO MSA not yet impl
        self.decode_iters = decode_iters
        self.device = device

    def load_code_from_alist(self, header_fn):
        params = get_ldpc_code_params(header_fn, compute_matrix=True)
        params['decode_algorithm'] = 'SPA'
        return params

    def zero_pad(self, x, modulo):
        B, _, L = x.size()
        if torch.numel(x[0]) % modulo != 0:
            n_pad = (modulo*L) - (torch.numel(x[0]) % (modulo*L))
            zero_pad = torch.zeros(B, n_pad, device=x.device).view(B, -1, L)
            padded_message = torch.cat((x, zero_pad), dim=1)
        else:
            padded_message = x
        return padded_message

    def encode(self, message_bits):
        codewords = torch.matmul(message_bits, self.G) % 2
        return codewords

    def decode(self, symbol_llr):
        # NOTE process batch by batch due to memory use
        batch_llr = torch.chunk(symbol_llr, chunks=1, dim=0)
        out_llr = []
        for b_llr in batch_llr:
            block_llr = torch.chunk(b_llr, chunks=2, dim=1)
            d_llr = []
            for llr_i in block_llr:
                decoded_bits, llr, _ = ldpc_bp_decode(llr_i, self.ldpc_design, self.H,
                                           self.decode_algorithm, self.decode_iters)
                d_llr.append(llr)

            d_llr = torch.cat(d_llr, dim=1)
            out_llr.append(d_llr)

        llr = torch.cat(out_llr, dim=0)
        return decoded_bits, llr
    

    
# Function for converting propabilities to LLRs
# I assume here that the Pr lookup table is a n*2 2D array
# For each symbol, I assume the first column is the Pr that it's 0, while the 2nd column is the Pr that it's a 1
# LLR_j = log(Pr(x_j = 1) / Pr(x_j = 0))
def LLR_convertion(probs, codelength):
    LLR_vec = torch.zeros(codelength)
    for i in range(codelength):
        LLR_vec[i] = torch.log(probs[i,1] / probs[i,0])
    return LLR_vec

