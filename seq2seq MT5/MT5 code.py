

import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
hostname = socket.gethostname()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')






PAD_token = 0
SOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)



# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    #如果前面有空格，则按照nfd标准化把uniclode码转化成ascii码
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) #我们可以在 for 语句后面跟上一个 if 判断语句，用于过滤掉那些不满足条件的结果项。
        if unicodedata.category(c) != 'Mn'  # unicodedata.category():把一个字符返回它在UNICODE里分类的类型  [Mn] Mark, Nonspacing 
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s



def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    filename = 'fren.train1.txt'
    lines = []
    with open(filename) as f:
        for line in f:
            lines.append(line.strip().replace("\n", ""))
#     lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('|||')] for l in lines]
#     pairs = [[s.strip() for s in l.split('|||')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs




MIN_LENGTH = 0
MAX_LENGTH = 30

def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH \
            and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs



def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    # pairs = [[normalize_string(s) for s in l.split('|||')] for l in lines]
    # input_lang=Lang(french)
    # output_lang=Lang(english)
    print("Read %s sentence pairs" % len(pairs))
    
#     pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('fren', 'train1', False)

print(random.choice(pairs))





keep_pairs = []

for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True
    
    for word in input_sentence.split(' '):
        if word not in input_lang.word2index:
            keep_input = False
            break

    for word in output_sentence.split(' '):
        if word not in output_lang.word2index:
            keep_output = False
            break

    # Remove if pair doesn't match input and output conditions
    if keep_input and keep_output:
        keep_pairs.append(pair)

print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
pairs = keep_pairs




# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]




# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq




def random_batch(batch_size):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
        
    return input_var, input_lengths, target_var, target_lengths





random_batch(2)[0].shape,random_batch(2)[2].shape




# ## The Encoder



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden
EncoderRNN(5,3,2)
#lengths.to("cpu")
data = torch.tensor([[1, 2, 0],[1, 2, 3]])
print(data.shape)
EncoderRNN(5,3,2).forward(input_seqs=data, input_lengths=[2,2,2])
print(torch.tensor([2,4]))


# In[ ]:





# In[ ]:





# ## Attention Decoder



class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size) # LSTM hidden size To 2*LSTM hidden size

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            # self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            self.v = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            #energy = hidden.dot(encoder_output)
            energy=torch.matmul(hidden,encoder_output.reshape(-1, 1))
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            #energy = hidden.dot(energy)
            energy=torch.matmul(hidden,energy.reshape(-1, 1))
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            # energy = self.v.dot(energy.squeeze(0))
            energy = self.v(energy.squeeze(0))
            # energy=torch.matmul(self.v,energy.reshape(-1, 1))
            return energy
Attn('general',10)
# Attn.score(hidden,outputs)


#
# The decoder's inputs are the last RNN hidden state $s_{i-1}$, last output $y_{i-1}$, and all encoder outputs $h_*$.
# 
# * embedding layer with inputs $y_{i-1}$
#     * `embedded = embedding(last_rnn_output)`
# * attention layer $a$ with inputs $(s_{i-1}, h_j)$ and outputs $e_{ij}$, normalized to create $a_{ij}$
#     * `attn_energies[j] = attn_layer(last_hidden, encoder_outputs[j])`
#     * `attn_weights = normalize(attn_energies)`
# * context vector $c_i$ as an attention-weighted average of encoder outputs
#     * `context = sum(attn_weights * encoder_outputs)`
# * RNN layer(s) $f$ with inputs $(s_{i-1}, y_{i-1}, c_i)$ and internal hidden state, outputting $s_i$
#     * `rnn_input = concat(embedded, context)`
#     * `rnn_output, rnn_hidden = rnn(rnn_input, last_hidden)`
# * an output layer $g$ with inputs $(y_{i-1}, s_i, c_i)$, outputting $y_i$
#     * `output = out(embedded, rnn_output, context)`

#


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        #self.max_length = max_length
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('general', hidden_size)
        self.lstm = nn.LSTM(hidden_size*2, hidden_size, n_layers, dropout=dropout_p, bidirectional=False)
        self.out = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # TODO: FIX BATCHING
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).unsqueeze(0)# .view(1, 1, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[0][-1].unsqueeze(0), encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.lstm(rnn_input, last_hidden)
        
        # Final output layer
        # output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 2)),dim=-1)
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
BahdanauAttnDecoderRNN(10,3,2)





from unicodedata import bidirectional


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.concat = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.lstm(embedded, last_hidden)
        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #encoder_outputs:(seq_len,bs,hs) 
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


# #no use at this code
# class Seq2SeqAttentionMechanism(nn.Module):
#
#     def __init__(self):
#         super(Seq2SeqAttentionMechanism,self).__init__()
#     def forward(self,decoder_state_t,encoder_states):   #decoder_state_t t时刻解码器状态 ,encoder_states一般不是流式，encoder接受的源句可以一次性拿到
#         bs,source_lenth,hidden_size=encoder_states.shape
#         decoder_state_t=decoder_state_t.unsqueeze(1)#bs*1*hidden_size
#         decoder_state_t=torch.tile(decoder_state_t,dims=(1,source_lenth,1))   #3d tensor  复制的和编码器长度一致
#         score = torch.sum(decoder_state_t*encoder_states,dim=-1)   #内积 #[bs,source_lenth]
#         attn_prob = F.softmax(score,dim=-1) #decoder state和encoder state状态关系的权重
#         context = torch.sum(attn_prob.unsqueeze(-1)*encoder_states,1) #用到了广播机制 #对时间维度进行求和 context = [bs,hidden_size] 第t时刻解码器所需要的上下文向量
#         return attn_prob,context
#
# class Seq2SeqDecoder(nn.Module):
#     def __init__(self,embedding_dim,hidden_size,num_classes,target_vocab_size,start_if,end_id):
#         super(Seq2SeqDecoder,self).__init__()
#         self.lstm_cell=torch.nn.LSTMCell(embedding_dim,hidden_size) #2D tensor
#
#         self.proj_layer = nn.Linear(hidden_size*2,num_classes)  #proj_layer为context vector 和 decoder sstate两个拼起来的大小，这里假设他俩大小一样所以*2
#         self.attention_mechanism=Seq2SeqAttentionMechanism() #attention实例化
#         self.num_classes= num_classes
#         self.embedding_table=torch.nn.Embedding(target_vocab_size, embedding_dim)
#         self.start_id = start_id
#         self.end_id = end_id
#
#     def forward(self,shifted_target_ids,encoder_states): #,shifted_target_ids第一个位置是start_id #encoder_states是完整的编码器输出的序列
#         #训练阶段调用,teacher_force model：依赖于真实的老师“target_ids”去指导
#         shifted_target = self.embedding_table(shifted_target_ids) #把二维变成3维度，包含embedding_vector
#         bs, target_length, embedding_dim  =shifted_target.shape
#         bs, source_lenth,hidden_length = encoder_states.shape
#         logits = torch.zeros(bs,target_length, self.num_classes)
#         probs= torch.zeros(bs,target_length,source_length)
#         for t in range(target_length) : #训练阶段target_length已经知道，可以用for循环去做，推理阶段不知道只能用我while循环去做
#             decoder_input_t=shifted_target[:,t,:] #[bs,embedding_dim]
#             if t == 0:
#                 h_t,c_t=self.lstm_cell(decoder_input_t) #不传ht和ct，有默认值
#             else:
#                 h_t,c_t=self.lstm_cell(decoder_input_t,(h_t,c_t)) # h_t是解码器当前的状态
#
#             attn_prob,context = self.attention_mechanism(h_t,encoder_states)# attn_prob是解码器状态对编码器每个位置上的一个注意力权重
#                                                                              #以及根据权重和编码器状态算出来的context向量
#                                                                             #  context向量大小为 bs*hidden_size 二维的tensor
#             decoder_output = torch.cat((context,h_t),-1)
#             logits[:,t,:]=self.proj_layer(decoder_output)
#             probs[:,t,:]=attn_prob
#         return probs,logits #logits整个解码器完整的分类的logits #probs解码器每一步的对编码器的权重  他俩都是整体的一个三维张量
#
#     def inference(self,encoder_states):  #推理的时候不知道target_id（真实）,用的是解码器上一刻预测的id作为下一刻的输入
#         #推理阶段
#         target_id = self.start_id
#         h_t=None
#         while True:
#             decoder[input_t]=self.embedding_table(target_id) #初始的解码器输入，是固定的，用全0向量也可以只要保证训练阶段和推理阶段第一时刻的解码器输入是一致的就可
#             if h_t is None:
#                 h_t,c_t = self.lstm_cell(decoder_input_t)
#             else:
#                 h_t,c_t = self.lstm_cell(decoder_input_t,(h_t,c_t))
#             attn_prob, context = self.attention+mechanism(h_t,encoder_states)
#
#             decoder_output=torch.cat((context,h_t),-1)
#             logits=self.proj_layer(decoder_output)#分类logits
#             target_id.torch.argmax(logits,-1) #上一时刻预测的id作为下一时刻解码器的输入，自回归
#             result.append(target_id)
#
#             #必须有终止条件
#             if torch.any(target_id == self.end_id): #由于是batch，因此target_id 是batch_size个，构造训练数据的时候要注意，要把每个句子最后一个位置填充end_id
#                 print("stop decoding")
#                 break
#         predicted_ids = torch.stack(result,dim=0)
#
#         return predicted_ids
#
#
# # ## Testing the models
# #
# # To make sure the encoder and decoder modules are working (and working together) we'll do a full test with a small batch.
#
# # In[18]:


small_batch_size = 3
input_batches, input_lengths, target_batches, target_lengths = random_batch(small_batch_size)

print('input_batches', input_batches.size()) # (max_len x batch_size)
print('target_batches', target_batches.size()) # (max_len x batch_size)





input_batches





small_hidden_size = 8
small_n_layers = 2

encoder_test = EncoderRNN(input_lang.n_words, small_hidden_size, small_n_layers)
decoder_test = BahdanauAttnDecoderRNN(small_hidden_size, output_lang.n_words, small_n_layers)


# To test the encoder, run the input batch through to get per-batch encoder outputs:

# In[21]:


encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)

print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size
print('encoder_hidden', encoder_hidden[0].size(), encoder_hidden[1].size()) # n_layers * 2 x batch_size x hidden_size


# Then starting with a SOS token, run word tokens through the decoder to get each next word token. Instead of doing this with the whole sequence, it is done one at a time, to support using it's own predictions to make the next prediction. This will be one time step at a time, but batched per time step. In order to get this to work for short padded sequences, the batch size is going to get smaller each time.

# In[22]:


max_target_length = max(target_lengths)

# Prepare decoder input and outputs
decoder_input = Variable(torch.LongTensor([SOS_token] * small_batch_size))
# decoder_hidden = encoder_hidden[:decoder_test.n_layers] # Use last (forward) hidden state from encoder
n_layers = encoder_test.n_layers
decoder_hidden = (encoder_hidden[0][-n_layers:,:,:],encoder_hidden[1][-n_layers:,:,:])
all_decoder_outputs = Variable(torch.zeros(max_target_length, small_batch_size, decoder_test.output_size))

print(decoder_input.shape)
# Run through decoder one time step at a time
for t in range(max_target_length):
    decoder_output, decoder_hidden, decoder_attn = decoder_test(
        decoder_input, decoder_hidden, encoder_outputs
    )
    all_decoder_outputs[t] = decoder_output # Store this step's outputs
    decoder_input = target_batches[t] # Next input is current target

# Test masked cross entropy loss
loss = masked_cross_entropy(
    all_decoder_outputs.transpose(0, 1).contiguous(),
    target_batches.transpose(0, 1).contiguous(),
    target_lengths
)
print('loss', loss.item())


# # Training



def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    # decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    n_layers = encoder.n_layers
    decoder_hidden = (encoder_hidden[0][-n_layers:,:,:],encoder_hidden[1][-n_layers:,:,:])

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()
    
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item(), ec, dc


# ## Running training



# Configure models
attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout = 0.5
batch_size = 50

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.001
decoder_learning_ratio = 5.0
n_epochs = 200
epoch = 0
plot_every = 1
print_every = 1
evaluate_every = 1000

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout=dropout)
decoder = BahdanauAttnDecoderRNN(hidden_size, output_lang.n_words, n_layers)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()


start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every





def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# # Evaluating the network



from queue import PriorityQueue
import operator
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # 注意这里是有惩罚参数的，参考恩达的 beam-search

    def __lt__(self, other):
        return self.leng < other.leng  # 这里展示分数相同的时候怎么处理冲突，具体使用什么指标，根据具体情况讨论

    def __gt__(self, other):
        return self.leng > other.leng


def beam_search_evaluate(input_seq, max_length=MAX_LENGTH):
    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    
    input_lengths = [len(input_seq.split(' '))]
    input_seqs = [indexes_from_sentence(input_lang, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    # Run through encoder
    encoder_output, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    n_layers = encoder.n_layers
    decoder_hidden = (encoder_hidden[0][-n_layers:,:,:],encoder_hidden[1][-n_layers:,:,:])
    
    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))
    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()
    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > max_length: break

        # fetch the best node
        score, n = nodes.get()
        # print('--best node seqs len {} '.format(n.leng))
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == EOS_token and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        decoder_output, decoder_hidden, _ = decoder(torch.LongTensor([decoder_input]), decoder_hidden, encoder_output)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(decoder_output, beam_width)
        indexes = indexes.squeeze()
        log_prob = log_prob.squeeze()
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[new_k]
            log_p = log_prob[new_k].item()

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)

        utterance = utterance[::-1]
        utterances.append(utterance)
    
    results = []
    for utterance in utterances:
        results.append([output_lang.index2word[int(i)] for i in utterance])
    return results


# In[27]:


def evaluate(input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq.split(' '))]
    input_seqs = [indexes_from_sentence(input_lang, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
        
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    n_layers = encoder.n_layers
    decoder_hidden = (encoder_hidden[0][-n_layers:,:,:],encoder_hidden[1][-n_layers:,:,:])
    
    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]




# We can evaluate random sentences from the training set and print out the input, target, and output to make some subjective quality judgements:



def evaluate_randomly():
    [input_sentence, target_sentence] = random.choice(pairs)
    evaluate_and_show_attention(input_sentence, target_sentence)


# # Visualizing attention


import io
import torchvision
from PIL import Image
import visdom

vis = visdom.Visdom()

def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})


# For a better viewing experience we will do the extra work of adding axes and labels:

# In[30]:


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(attentions.numpy())
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    plt.show()
    plt.close()


# In[31]:


def evaluate_and_show_attention(input_sentence, target_sentence=None):
    output_words, attentions = evaluate(input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)
    
    show_attention(input_sentence, output_words, attentions)
    
    # Show input, target, output text in visdom
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    vis.text(text, win=win, opts={'title': win})


# # Putting it all together
# 



print(encoder)
print(decoder)


# In[ ]:


# Begin!
ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1
    
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)

    # Run the train function
    loss, ec, dc = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec
    dca += dc
    
    # job.record(epoch, loss)

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        
    if epoch % evaluate_every == 0:
        evaluate_randomly()

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        
        # TODO: Running average helper
        ecs.append(eca / plot_every)
        dcs.append(dca / plot_every)
        ecs_win = 'encoder grad (%s)' % hostname
        dcs_win = 'decoder grad (%s)' % hostname
        vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
        vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
        eca = 0
        dca = 0


# In[ ]:


#save model to file
torch.save(encoder,"./encoder.pt")
torch.save(decoder,"./decoder.pt")

#load mode from filere
encoder = torch.load("./encoder.pt")
decoder = torch.load("./decoder.pt")




def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

show_plot(plot_losses)


# In[ ]:


#output_words, attentions = evaluate("depend vie dispo plac .")
output_words, attentions = evaluate("vous n etes pas seule .")
print(attentions.numpy())
plt.matshow(attentions.numpy())
show_plot_visdom()


# In[ ]:


evaluate_and_show_attention("vous n etes pas seule .")




def read_langs_test(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    filename = 'fren.test1.txt'
    lines = []
    with open(filename) as f:
        for line in f:
            lines.append(line.strip().replace("\n", ""))
#     lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('|||')] for l in lines]
#     pairs = [[s.strip() for s in l.split('|||')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
    else:
        input_lang = Lang(lang1)

    return input_lang, pairs



def prepare_data_test(lang1_name, lang2_name, reverse=False):
    input_lang, pairs = read_langs_test(lang1_name, lang2_name, reverse)
    # pairs = [[normalize_string(s) for s in l.split('|||')] for l in lines]
    # input_lang=Lang(french)
    # output_lang=Lang(english)
    print("Read %s sentence pairs" % len(pairs))
    
#     pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])


    return input_lang, pairs



# In[ ]:


input_lang, pairs = prepare_data_test('fren', 'test1', False)

print(random.choice(pairs))


# In[ ]:


out = []
for pair in pairs:
    output_words, _ = evaluate(pair[0])
    out.append(output_words)

print(len(out))
print(random.choice(out))


# In[ ]:


from tqdm import auto
out = []
for pair in auto.tqdm(pairs):
    output_topk_words = beam_search_evaluate(pair[0]) # output [[token1,token2,...],[...],...]
    out.append(output_topk_words)

print(len(out))
print(out[0])




with open('out1.txt','w',encoding="utf-8") as f:
    for result in out:
        f.write(' '.join(result)+'\n')





