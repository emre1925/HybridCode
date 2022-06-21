import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from utils import *
from nn_layers import *
from parameters import *
import numpy as np






##################### Author @Emre Ozfatura #####################################

#### Import code headers ###
header_fileName = "Headers/BCH_63_45.alist"


#####################Neurol Encoder-Decoder no feedback

################################## Distributed training approach #######################################################



def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

################################ Prepare optimizer ######################################################################









########################## This is the overall AutoEncoder model ########################


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.ind1 = [0,4,8,12,16,20]
        self.ind2 = [22,18,14,10,6,2]
        self.args = args
        ################## We use learnable positional encoder which can be removed later ######################################
        self.pe = PositionalEncoder_fixed()
        ########################################################################################################################
        self.Tmodel = BERT('trx', 2**args.block_size, args.block_size, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.parity_pb, args.custom_attn,args.multclass)
        # mod, input_size, block_size, d_model, N, heads, dropout, outsize, custom_attn=True, multclass = False
        self.Rmodel = BERT('rec', args.parity_pb, args.block_size, args.d_model_trx, args.N_trx+1, args.heads_trx, args.dropout, 2**args.block_size ,args.custom_attn,args.multclass)
        self.Rmodelf = BERT('rec2', args.parity_pb + args.block_size, args.block_size, args.d_model_trx, args.N_trx+1, args.heads_trx, args.dropout, 2**args.block_size ,args.custom_attn,args.multclass)
        ########## Power Reallocation as in deepcode work ###############
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)

    def power_constraint(self, inputs): # Normalize through batch dimension
        this_mean = torch.mean(inputs, dim=0)
        this_std  = torch.std(inputs, dim=0)
        outputs   = (inputs - this_mean)*1.0/this_std
        return outputs

    ########### IMPORTANT ##################
    # We use modulated bits at encoder
    #######################################
    def forward(self, bVec, fwd_noise_par, table, isTraining = 1):
        ###############################################################################################################################################################
       ############# Generate the output ###################################################
        output = self.Tmodel(bVec, None, self.pe)
        parity = self.power_constraint(output)
        received =  parity + fwd_noise_par
        # ------------------------------------------------------------ receiver
        decSeq = self.Rmodel(received, None, self.pe) # Decode the sequence
        belief = torch.matmul(decSeq, table)
        ############################# consensus with repetiton code ########################
        belief_revised = (belief[:,0,:].unsqueeze(dim=1) + (1-belief[:,22,:].unsqueeze(dim=1)))/2
        for i in range (1,23):
            if np.mod(i, 4) == 0:
                belief_revised = torch.cat([belief_revised, (belief[:,i,:].unsqueeze(dim=1) + (1-belief[:,22-i,:].unsqueeze(dim=1)))/2],dim=1)
            elif np.mod(i, 4) == 2:
                belief_revised = torch.cat([belief_revised, 1-((belief[:,22-i,:].unsqueeze(dim=1) + (1-belief[:,i,:].unsqueeze(dim=1)))/2)],dim=1)
            else:
                belief_revised = torch.cat([belief_revised, belief[:,i,:].unsqueeze(dim=1)],dim=1)
        received_prior = torch.cat([received,belief],dim=2)
        ################ Repeat the process 1 ##################################################
        decSeq = self.Rmodelf(received_prior, None, self.pe) # Decode the sequence
        belief = torch.matmul(decSeq, table)
        ############################# consensus with repetiton code ########################
        belief_revised = (belief[:,0,:].unsqueeze(dim=1) + (1-belief[:,22,:].unsqueeze(dim=1)))/2
        for i in range (1,23):
            if np.mod(i, 4) == 0:
                belief_revised = torch.cat([belief_revised, (belief[:,i,:].unsqueeze(dim=1) + (1-belief[:,22-i,:].unsqueeze(dim=1)))/2],dim=1)
            elif np.mod(i, 4) == 2:
                belief_revised = torch.cat([belief_revised, 1-((belief[:,22-i,:].unsqueeze(dim=1) + (1-belief[:,i,:].unsqueeze(dim=1)))/2)],dim=1)
            else:
                belief_revised = torch.cat([belief_revised, belief[:,i,:].unsqueeze(dim=1)],dim=1)
        received_prior = torch.cat([received,belief],dim=2)
        ################ Repeat the process 2 ##################################################
        decSeq = self.Rmodelf(received_prior, None, self.pe) # Decode the sequence
        belief = torch.matmul(decSeq, table)
        ############################# consensus with repetiton code ########################
        belief_revised = (belief[:,0,:].unsqueeze(dim=1) + (1-belief[:,22,:].unsqueeze(dim=1)))/2
        for i in range (1,23):
            if np.mod(i, 4) == 0:
                belief_revised = torch.cat([belief_revised, (belief[:,i,:].unsqueeze(dim=1) + (1-belief[:,22-i,:].unsqueeze(dim=1)))/2],dim=1)
            elif np.mod(i, 4) == 2:
                belief_revised = torch.cat([belief_revised, 1-((belief[:,22-i,:].unsqueeze(dim=1) + (1-belief[:,i,:].unsqueeze(dim=1)))/2)],dim=1)
            else:
                belief_revised = torch.cat([belief_revised, belief[:,i,:].unsqueeze(dim=1)],dim=1)
        received_prior = torch.cat([received,belief],dim=2)
        ################ Final ################
        decSeqf = self.Rmodelf(received_prior, None, self.pe) # Decode the sequence
        prediction = F.softmax(decSeqf, dim=-1)
        return prediction




############################################################################################################################################################################








def train_model(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    start = time.time()
    epoch_loss_record = []
    flag = 0
    if args.block_size == 3:
        map_vec = torch.tensor([1,2,4])# maping block of bits to class label
    else:
        map_vec = torch.tensor([1,2])# maping block of bits to class label
    # in each run, randomly sample a batch of data from the training dataset
    ################################### Vector embedding ###################################
    A_blocks = torch.tensor([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],requires_grad=False).float()
    ################################### Distance based vector embedding ####################
    Embed = torch.zeros(8,args.batchSize, args.numb_block+6, 8)
    refind=[15,12,9,6,3,0]
    for i in range(8):
        embed = torch.zeros(8)
        for j in range(8):
            embed[j] = torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))
        Embed[i,:,:,:]= embed.repeat(args.batchSize, args.numb_block+6, 1)
    
########################################################################################################### 
    numBatch = (1000 * args.totalbatch) + 1 # Total number of batches
    for eachbatch in range(numBatch):
        ############################## Here we directy generate symbols ###################################
        bVec = torch.randint(0, 8, (args.batchSize, args.numb_block, 1),requires_grad=False)
        bVec_ref = bVec[:,refind,:]
        bVec_ref_inv = 7-bVec_ref
        ############################## This part is inverse repetition code #######################################
        bVec_hyb = torch.zeros((args.batchSize, args.numb_block+6,1),requires_grad=False) # generated data in terms of distance embeddings
        for i in range (6):
            if i==5:
                bVec_hyb[:,i*4:i*4+1,:] = bVec[:,i*3:i*3+1,:] 
                bVec_hyb[:,i*4+2,:] = bVec_ref_inv[:,i,:] 
            else:
                bVec_hyb[:,i*4:i*4+1,:] = bVec[:,i*3:i*3+1,:] 
                bVec_hyb[:,i*4+2,:] = bVec_ref_inv[:,i,:] 
                bVec_hyb[:,i*4+3,:] = bVec[:,i*3+3,:] 
        ############################## Generated data in the form of vector embeddings #################################
        bVecr = torch.zeros((args.batchSize, args.numb_block+6,8), requires_grad=False) # generated data in terms of distance embeddings
        for i in range(8):
            mask = (bVec_hyb == i).long()
            bVecr= bVecr + (mask * Embed[i,:,:,:])
        #################################### Generate noise sequence ##################################################
        ###############################################################################################################
        ###############################################################################################################
        ################################### Curriculum learning strategy ##############################################
        if eachbatch < args.core * 80000:
           snr=4* (1-eachbatch/(args.core * 80000))+ (eachbatch/(args.core * 80000)) * args.snr
        else:
           snr=args.snr
        ################################################################################################################
        stdn = 10 ** (-snr * 1.0 / 10 / 2) #forward snr
        # Noise values for the parity bits
        fwd_noise_par = torch.normal(0, std=stdn, size=(args.batchSize, args.numb_block+6, args.parity_pb), requires_grad=False)
        ############## feed into model to get predictions##########################
        preds = model(bVecr.to(args.device), fwd_noise_par.to(args.device), A_blocks.to(args.device), isTraining=1)
        ############## Optimization ###############################################
        args.optimizer.zero_grad()
        ################################ loss #############################
        ys = bVec_hyb.long().contiguous().view(-1)
        preds = preds.contiguous().view(-1, preds.size(-1))
        preds = torch.log(preds)
        loss = F.nll_loss(preds, ys.to(args.device))########################## This should be binary cross-entropy loss
        loss.backward()
        ####################### Gradient Clipping optional ###########################
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        ############################ Schedule the learning rate ##################################################
        args.optimizer.step()
        if args.use_lr_schedule:
            args.scheduler.step()
        #############################################################################
        ################################ Observe test accuracy##############################
        with torch.no_grad():
            probs, decodeds = preds.max(dim=1)
            succRate = sum(decodeds == ys.to(args.device)) / len(ys)
            print('NoFB_nc','Idx,lr,BS,loss,BER,num=', (
            eachbatch, args.lr, args.batchSize, round(loss.item(), 5), round(1 - succRate.item(), 6),
            sum(decodeds != ys.to(args.device)).item()))
        ####################################################################################
        if np.mod(eachbatch, 10000) == 0:
            epoch_loss_record.append(loss.item())
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/model_weights' + str(eachbatch)
            torch.save(model.state_dict(), saveDir)


def EvaluateNets(model, args):
    checkpoint = torch.load(args.saveDir)
    # # ======================================================= load weights
    model.load_state_dict(checkpoint)
    model.eval()

    ################################### Vector embedding ###################################
    A_blocks = torch.tensor([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],requires_grad=False).float()
    ################################### Distance based vector embedding ####################
    Embed = torch.zeros(8,args.batchSize, args.numb_block+6, 8)
    refind=[15,12,9,6,3,0]
    for i in range(8):
        embed = torch.zeros(8)
        for j in range(8):
            embed[j] = torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))
        Embed[i,:,:,:]= embed.repeat(args.batchSize, args.numb_block+6, 1)

    args.numTestbatch = 100000

    # failbits = torch.zeros(args.K).to(args.device)
    bitErrors = 0
    pktErrors = 0
    for eachbatch in range(args.numTestbatch):
        bVec = torch.randint(0, 8, (args.batchSize, args.numb_block, 1),requires_grad=False)
        bVec_ref = bVec[:,refind,:]
        bVec_ref_inv = 7-bVec_ref
        ############################## This part is inverse repetition code #######################################
        bVec_hyb = torch.zeros((args.batchSize, args.numb_block+6,1),requires_grad=False) # generated data in terms of distance embeddings
        for i in range (6):
            if i==5:
                bVec_hyb[:,i*4:i*4+1,:] = bVec[:,i*3:i*3+1,:] 
                bVec_hyb[:,i*4+2,:] = bVec_ref_inv[:,i,:] 
            else:
                bVec_hyb[:,i*4:i*4+1,:] = bVec[:,i*3:i*3+1,:] 
                bVec_hyb[:,i*4+2,:] = bVec_ref_inv[:,i,:] 
                bVec_hyb[:,i*4+3,:] = bVec[:,i*3+3,:] 
        ############################## Generated data in the form of vector embeddings #################################
        bVecr = torch.zeros((args.batchSize, args.numb_block+6,8), requires_grad=False) # generated data in terms of distance embeddings
        for i in range(8):
            mask = (bVec_hyb == i).long()
            bVecr= bVecr + (mask * Embed[i,:,:,:])
        stdn = 10 ** (-args.snr * 1.0 / 10 / 2) #forward snr
        # Noise values for the parity bits
        fwd_noise_par = torch.normal(0, std=stdn, size=(args.batchSize, args.numb_block+6, args.parity_pb), requires_grad=False)

        # feed into model to get predictions
        with torch.no_grad():
            preds = model(bVecr.to(args.device), fwd_noise_par.to(args.device), A_blocks.to(args.device),isTraining=0)
            preds1 = preds.contiguous().view(-1, preds.size(-1))
            ys = bVec_hyb.contiguous().view(-1)
            probs, decodeds = preds1.max(dim=1)
            decisions = decodeds != ys.to(args.device)
            bitErrors += decisions.sum()
            BER = bitErrors / (eachbatch + 1) / args.batchSize / (args.K+18)
            pktErrors += decisions.view(args.batchSize, 23).sum(1).count_nonzero()
            PER = pktErrors / (eachbatch + 1) / args.batchSize
            print('num, BER, errors, PER, errors = ', eachbatch, round(BER.item(), 10), bitErrors.item(),
                  round(PER.item(), 10), pktErrors.item(), )

    BER = bitErrors.cpu() / (args.numTestbatch * args.batchSize * args.K)
    PER = pktErrors.cpu() / (args.numTestbatch * args.batchSize)
    print(BER)
    print("Final test BER = ", torch.mean(BER).item())
    pdb.set_trace()


if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    ########### path for saving model checkpoints ################################
    args.saveDir = 'weights/model_weights100000'  # path to be saved to
    ################## Model size part ###########################################
    args.d_model_trx = args.heads_trx * args.d_k_trx # total number of features
    ##############################################################################
    args.total_iter = 101000
    # ======================================================= Initialize the model
    model = AE(args).to(args.device)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    # ======================================================= run
    if args.train == 1:
        if args.opt_method == 'adamW':
        	args.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        elif args.opt_method == 'lamb':
        	args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        else:
        	args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
        	lambda1 = lambda epoch: (1-epoch/args.total_iter)
        	args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)



        if 0:
            checkpoint = torch.load(args.saveDir)
            model.load_state_dict(checkpoint)
            print("================================ Successfully load the pretrained data!")

        train_model(model, args)
    else:
        EvaluateNets(model, args)
