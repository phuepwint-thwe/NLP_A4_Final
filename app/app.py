import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from random import seed, shuffle, random, randint
from sklearn.metrics.pairwise import cosine_similarity

# Define vocabulary and mappings
from datasets import load_dataset
dataset = load_dataset('bookcorpus', split='train[:1%]')
dataset = dataset.select(range(100000))
sentences = dataset['text']
text = [x.lower() for x in sentences] #lower case
text = [re.sub("[.,!?\\-]", '', x) for x in text]


word_list = list(set(" ".join(text).split()))
word2id   = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

for i, w in enumerate(word_list):
    word2id[w] = i + 4 #reserve the first 0-3 for CLS, PAD
    id2word    = {i:w for i, w  in enumerate(word2id)}
    vocab_size = len(word2id)
    
token_list = list()
for sentence in text:
    arr = [word2id[word] for word in sentence.split()]
    token_list.append(arr)

batch_size = 10
max_len    = 1000 #maximum length that my transformer will accept.....all sentence will be padded to this length
max_mask   = 5    # maximum number of masked tokens
d_model    = 768  # embedding dimension
n_layers   = 6
n_heads    = 8
d_ff       = d_model * 4
d_k = d_v  = 64
n_segments = 2

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k):
    # Create mask for PAD tokens (assumed to be 0)
    pad_attn_mask = seq_k.eq(0).unsqueeze(1)  # shape: [batch, 1, len_k]
    return pad_attn_mask.expand(seq_q.size(0), seq_q.size(1), seq_k.size(1))

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc  = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual = Q
        batch_size = Q.size(0)
        # Project and split into multiple heads
        q = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q, k, v, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, attn_mask):
        enc_outputs, attn = self.self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask)
        enc_outputs = self.ffn(enc_outputs)
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        # For next sentence prediction (NSP)
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)
        # For masked language modeling (LM)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Decoder shares weights with token embedding
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.weight = self.embedding.tok_embed.weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, _ = layer(output, attn_mask)
        # NSP: use [CLS] token (first token)
        h_pooled = self.activ(self.fc(output[:, 0]))
        logits_nsp = self.classifier(h_pooled)
        # LM: gather positions for masked tokens
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_nsp

def mean_pool(token_embeds, attention_mask):
    # Expand attention mask to match embed dimensions and compute mean
    mask = attention_mask.unsqueeze(-1).float()
    pooled = torch.sum(token_embeds * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return pooled

def tokenize_sentence_model1(sentence_a, sentence_b):
    seed(55)  # for reproducibility
    max_seq_length = max_len

    # Convert words to token ids; handle out-of-vocabulary words
    tokens_a = [word2id[word] if word in word_list else len(word_list) for word in sentence_a.split()]
    tokens_b = [word2id[word] if word in word_list else len(word_list) for word in sentence_b.split()]

    # Add special tokens
    input_ids_a = [word2id['[CLS]']] + tokens_a + [word2id['[SEP]']]
    input_ids_b = [word2id['[CLS]']] + tokens_b + [word2id['[SEP]']]

    # Create segment ids (here we use all zeros)
    segment_ids = [0] * max_seq_length

    def apply_masking(input_ids):
        num_to_mask = min(max_mask, max(1, int(round(len(input_ids) * 0.15))))
        candidates = [i for i, token in enumerate(input_ids) if token not in (word2id['[CLS]'], word2id['[SEP]'])]
        shuffle(candidates)
        masked_pos = []
        for pos in candidates[:num_to_mask]:
            masked_pos.append(pos)
            rand_val = random()
            if rand_val < 0.1:
                # Replace with random token
                input_ids[pos] = word2id[id2word[randint(0, vocab_size - 1)]]
            elif rand_val < 0.8:
                input_ids[pos] = word2id['[MASK]']
            # else, leave the token unchanged
        # Pad masked positions if necessary
        masked_pos += [0] * (max_mask - len(masked_pos))
        return masked_pos

    masked_pos_a = apply_masking(input_ids_a.copy())
    masked_pos_b = apply_masking(input_ids_b.copy())

    # Pad sequences to max_seq_length
    input_ids_a = (input_ids_a[:max_seq_length] + [0] * max_seq_length)[:max_seq_length]
    input_ids_b = (input_ids_b[:max_seq_length] + [0] * max_seq_length)[:max_seq_length]
    attention_a = [1 if token != 0 else 0 for token in input_ids_a]
    attention_b = [1 if token != 0 else 0 for token in input_ids_b]

    return {
        "premise_input_ids": [input_ids_a],
        "premise_pos_mask": [masked_pos_a],
        "hypothesis_input_ids": [input_ids_b],
        "hypothesis_pos_mask": [masked_pos_b],
        "segment_ids": [segment_ids],
        "attention_premise": [attention_a],
        "attention_hypothesis": [attention_b],
    }

def predict_nli_and_similarity(model, sentence_a, sentence_b, device):
    # Tokenize input sentences
    inputs = tokenize_sentence_model1(sentence_a, sentence_b)
    
    # Convert lists to tensors and move to device
    premise_ids = torch.tensor(inputs['premise_input_ids']).to(device)
    pos_mask_a = torch.tensor(inputs['premise_pos_mask']).to(device)
    attention_a = torch.tensor(inputs['attention_premise']).to(device)
    hypothesis_ids = torch.tensor(inputs['hypothesis_input_ids']).to(device)
    pos_mask_b = torch.tensor(inputs['hypothesis_pos_mask']).to(device)
    attention_b = torch.tensor(inputs['attention_hypothesis']).to(device)
    segments = torch.tensor(inputs['segment_ids']).to(device)
    
    # Get embeddings from model
    model.eval()
    with torch.no_grad():
        u, _ = model(premise_ids, segments, pos_mask_a)
        v, _ = model(hypothesis_ids, segments, pos_mask_b)
    u_mean = mean_pool(u, attention_a)
    v_mean = mean_pool(v, attention_b)
    
    # Compute cosine similarity
    u_np = u_mean.cpu().numpy().reshape(1, -1)
    v_np = v_mean.cpu().numpy().reshape(1, -1)
    similarity_score = cosine_similarity(u_np, v_np)[0, 0]
    
    # NLI prediction using a classifier head:
    # Concatenate u, v, and |u-v|
    uv_abs = torch.abs(u_mean - v_mean)
    features = torch.cat([u_mean, v_mean, uv_abs], dim=-1)
    
    # Global classifier_head: adjust dimensions if needed (here 768*3 = 2304)
    global classifier_head
    with torch.no_grad():
        logits = classifier_head(features)
        probs = F.softmax(logits, dim=-1)
    # Map predictions to labels (update order as desired)
    labels = ["Entailment", "Neutral", "Contradiction"]
    nli_result = labels[torch.argmax(probs).item()]
    
    return {"similarity_score": similarity_score, "nli_label": nli_result}

@st.cache(allow_output_mutation=True)
def load_bert_model():
    model = BERT()
    model.load_state_dict(torch.load('models/BERT-model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache(allow_output_mutation=True)
def load_sbert_model():
    model = BERT()  # Replace with your SBERT class if different
    model.load_state_dict(torch.load('models/SBERT-model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

# Define and initialize the classifier head (assumes d_model=23069)
classifier_head = torch.nn.Linear(23069 * 3, 3)

# ------------------------------------------------
# Streamlit App UI
# ------------------------------------------------

st.title("NLI and Similarity Prediction App")

st.write("Enter two sentences below to get the cosine similarity and NLI prediction.")

sentence_a = st.text_input("Sentence A:")
sentence_b = st.text_input("Sentence B:")

if st.button("Predict"):
    if sentence_a and sentence_b:
        # Load your BERT model (for NLI prediction)
        bert_model = load_bert_model()
        # (Optionally, you can load the SBERT model if you want to use it for similarity)
        # sbert_model = load_sbert_model()
        
        results = predict_nli_and_similarity(bert_model, sentence_a, sentence_b, device='cpu')
        similarity = results["similarity_score"]
        nli_prediction = results["nli_label"]
        
        st.write(f"**Cosine Similarity:** {similarity:.4f}")
        st.write(f"**NLI Prediction:** {nli_prediction}")
    else:
        st.error("Please enter both sentences.")
