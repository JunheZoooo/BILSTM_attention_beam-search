# BILSTM_attention_beam-search
![626831319641906050](https://user-images.githubusercontent.com/114964497/200099589-4ed3d61d-dc18-4d21-ab8a-78b47ace5a73.jpg)



# ENCODER:
(max_seq_len,B,src_voc_size)
embed(src_voc_size,embedding_size)
(max_seq_len,B,embedding_size)
LSTM(embedding_size,hidden_size,nlayer,bidirectional=True)
enc_out:(max_seq_len,B,hidden_size), enc_last_hidden:((2*nlayer,B,hidden_size),(2*nlayer,B,hidden_size))

# DECODER:
Attention(enc_last_hidden,enc_out)
attn_weights:(1,B,hidden_size)
context = attn_weights*enc_out
context:(1,B,hidden_size)

dec_input:(1,B,tgt_voc_size)
embed(tgt_voc_size,embedding_size)
word_emb:(1,B,embedding_size)
(lstm)rnn_input:concat(word_emb,context)
(lstm)rnn_input:(1,B,(hidden_size+embedding_size))
LSTM((hidden_size+embedding_size),hidden_size,nlayer,bidirectional=False)
dec_out:(1,B,hidden_size), dec_last_hidden:((nlayer,B,hidden_size),(nlayer,2,hidden_size))
concat(dec_out, context)
(1,B,hidden_size+hidden_size)
linear(hidden_size+hidden_size,tgt_voc_size)
(1,B,tgt_voc_size)
# SOFTMAX
(1,B,tgt_voc_size)





