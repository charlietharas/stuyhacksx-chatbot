"""# Sequence to Sequence

Vectorizing the dictionary of distinct words as a Vocabulary object, grabbing specialized conversation pairs, and training a model to respond. This is the core of the project, inspired by and heavily relying on content from [here](https://medium.com/swlh/end-to-end-chatbot-using-sequence-to-sequence-architecture-e24d137f9c78).
"""

import discord
import argparse
import unicodedata
import re
import random
import torch
from torch import nn
import itertools
import os
from tensordash.torchdash import Torchdash
import time
import datetime
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--train_file", help="CSV file used for training.", required=True)
parser.add_argument("-lf", "--load_file", help=".tar model save file.", required=True)
parser.add_argument("-u", "--user", help="User to mimic.", required=True)
parser.add_argument("-t", "--discord_bot_token", help="The token to your discord bot.", required=True)
parser.add_argument("-te", "--tensordash_email", help="Email for streaming model progress to Tensordash app. Leave blank if not using.")
parser.add_argument("-tp", "--tensordash_password", help="Password for streaming model progress to Tensordash app. Leave blank if not using.")
parser.add_argument("-lr", "--learning_rate", help="Learning rate Alpha of model. Default is 0.0001.", default=0.0001, type=float)
parser.add_argument("-dlr", "--decoder_learning", help="Decoder learning rate multiplier. Default is 5.0.", default=5.0, type=float)
parser.add_argument("-hidden", "--hidden_layer_size", help="Size of hidden layer. Default is 512.", default=512, type=int)
parser.add_argument("-e_layers", "--encoder_layers", help="Number of layers in the encoder. Default is 2.", default=2, type=int)
parser.add_argument("-d_layers", "--decoder_layers", help="Number of layers in the decoder. Default is 2.", default=2, type=int)
parser.add_argument("-drop", "--dropout_rate", help="Rate of dropout of the model. Default is 0.1.", default=0.1, type=float)
parser.add_argument("-attn_type", "--attention_model_type", help="Customized attention model type. Default is dot.", default='dot')

args = parser.parse_args()
sel_user = args.user
filename = args.train_file
tensordash_user = args.tensordash_email
tensordash_pass = args.tensordash_password
TOKEN = args.discord_bot_token

# data feature processing
data = pd.read_csv(filename)
authors = data['Author']
content = data['Content']
time_data = data['Date']
time_diff_list = []
conv_id_list = []
conv_id_list.append(0)
is_custom_user = []

def convert_datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str, "%d-%b-%y %I:%M %p").timestamp()

for i in range(1, len(time_data)):
    time_diff_to_app = (convert_datetime(time_data[i])-convert_datetime(time_data[i-1]))/60
    time_diff_list.append(time_diff_to_app)
    if time_diff_to_app >= 30:
        conv_id_list.append(conv_id_list[-1]+1)
    else:
        conv_id_list.append(conv_id_list[-1])
        
time_diff = pd.Series(time_diff_list)
conv_id = pd.Series(conv_id_list)

# model attributes have to be redefined, reusing code lines
# defining how a vocabulary object is set up
# relies on running above code to get count of distinct words as word_count
PAD = 0
SRT = 1
END = 2

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {PAD: "PAD", SRT: "SOS", END: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWordNoContext(word)

    def addWordNoContext(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.word_to_count[word] = 1
            self.index_to_word[self.num_words] = word
            self.num_words += 1
        else:
            self.word_to_count[word] += 1

    def addWord(self, word, index, count):
        self.word_to_index[word] = index
        self.word_to_count[word] = count
        self.index_to_word[index] = word
        self.num_words += 1

# functions to fix bad characters and clean up messages, optimizing convergence
def fixASCII(string):
    return ''.join(
        c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn'
    )

def fixString(string):
    string = fixASCII(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string

"""## Generating Sentence Pair Objects

Various methods of generating objects for sentence pair objects for training the model.
This section will also build specific vocabulary objects for each distinct conversation filter.

1.   "Less dumb" grabber between selected user for training and any other user. Considers only previous lines, offers little context, and scans the entire corpus.

"""

# "less dumb" grabber: builds pairs out of anyone talking to user
pairs = []
vocabulary = Vocabulary("Less dumb 2-user grabber")
for i in range(1, len(content)):
    if authors[i] == sel_user and [authors[i-1]] != sel_user:
        try:
            curr_cont = fixString(content[i])
            prev_cont = fixString(content[i-1])
            pairs.append([prev_cont, curr_cont])
            vocabulary.addSentence(curr_cont)
            vocabulary.addSentence(prev_cont)
        except:
            continue

print("Discriminant with any-user basic filter grabbed", len(pairs), "distinct pairs across entire corpus.")
print("Corresponding Vocabulary object with", vocabulary.num_words, "distinct valid words.")

"""## Data Preparation
Preparing batches for use in the model.
"""

# utility functions
# multi-grabs indexes from vocabulary
def getIndexesFromSent(voc, sent):
    return [voc.word_to_index[word] for word in sent.split(' ')] + [END]

# generating padding
def genPadding(batch, fillvalue=PAD):
    return list(itertools.zip_longest(*batch, fillvalue=fillvalue))

# returns binary matrix adjusting for padding
def binaryMatrix(batch, value=PAD):
    matrix = []
    for i, seq in enumerate(batch):
        matrix.append([])
        for token in seq:
            if token == PAD:
                matrix[i].append(0)
            else:
                matrix[i].append(1)
    return matrix

# padding functions
# return input tensor and corresponding lengths
def inputVariable(batch, voc):
    idxs_batch = [getIndexesFromSent(voc, sentence) for sentence in batch]
    lengths = torch.tensor([len(indexes) for indexes in idxs_batch])
    padded_list = genPadding(idxs_batch)
    padded_variable = torch.LongTensor(padded_list)
    return padded_variable, lengths

# return target tensor, padding mask, and maximum length
def outputVariable(batch, voc):
    idxs_batch = [getIndexesFromSent(voc, sentence) for sentence in batch]
    max_len = max([len(indexes) for indexes in idxs_batch])
    padded_list = genPadding(idxs_batch)
    mask = binaryMatrix(padded_list)
    mask = torch.ByteTensor(mask)
    padded_variable = torch.LongTensor(padded_list)
    return padded_variable, mask, max_len

# converts batch into train data
def batch_to_data(voc, batch):
    batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch = []
    output_batch = []
    for pair in batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inpt, lengths = inputVariable(input_batch, voc)
    output, mask, max_len = outputVariable(output_batch, voc)
    return inpt, lengths, output, mask, max_len

# example
batches = batch_to_data(vocabulary, [random.choice(pairs) for i in range(5)])
input_var, lengths, target_var, mask, max_len = batches

"""## The Model
The model in this case revolves around 3 layers

1.   An encoder to losslessly vectorize words into trainable binary sequences (for this we use a bidirectional GRU).
2.   An attention layer prioritizes different parts of sentences for "understanding." For this we use a Luong attention layer.
3.   A decoder to convert the model's inner "thoughts" into output for the user!
"""

if (tensordash_user is not None and tensordash_pass is not None):
    histories = Torchdash(ModelName="Chatbot", email=tensordash_user, password=tensordash_pass)

# encoder
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout), bidirectional=True)

    def forward(self, input_sequence, input_lengths, hidden=None):
        embedded = self.embedding(input_sequence)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

# attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not a valid attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()

        return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

# decoder
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=(0 if self.n_layers == 1 else dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.attn = Attn(self.attn_model, self.hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = nn.functional.softmax(output, dim=1)
        return output, hidden

# training functions
device = torch.device("cpu")

# searcher
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_sequence, input_length, max_len):
        encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SRT
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for i in range(max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores


hidden_size = args.hidden_layer_size
encoder_n_layers = args.encoder_layers
decoder_n_layers = args.decoder_layers
dropout = args.dropout_rate
attn_model = args.attention_model_type
alpha = args.learning_rate
decoder_learning = args.decoder_learning

embedding = nn.Embedding(vocabulary.num_words, hidden_size)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocabulary.num_words, decoder_n_layers, dropout)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=alpha)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=alpha * decoder_learning)

# loading models
load_filename = args.load_file
checkpoint = torch.load(load_filename)
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
vocabulary.__dict__ = checkpoint['voc_dict']
embedding.load_state_dict(embedding_sd)
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
encoder_optimizer.load_state_dict(encoder_optimizer_sd)
decoder_optimizer.load_state_dict(decoder_optimizer_sd)
encoder.to(device)
decoder.to(device)

# evaluation
def evaluate(searcher, voc, sent):
    idxs_batch = [getIndexesFromSent(voc, sent)]
    lengths = torch.tensor([len(indexes) for indexes in idxs_batch])
    input_batch = torch.LongTensor(idxs_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, 12)
    decoded_words = [voc.index_to_word[token.item()] for token in tokens]
    return decoded_words

def do_evaluate(searcher, voc, content):
    input_sent = content
    input_sent = fixString(input_sent)
    outputs = evaluate(searcher, voc, input_sent)
    outputs[:] = [x for x in outputs if not (x=='EOS' or x=='PAD')]
    return ' '.join(outputs)

searcher = GreedySearchDecoder(encoder, decoder)

"""# Discord Implementation
Code for running a discord bot with this model.
"""

client = discord.Client()

@client.event
async def on_ready():
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name='$help'))
    print('Logged on as user {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
        
    if message.content.startswith('$hello'):
        await message.channel.send('Hello! I am online and functional!')
        
    if message.content.startswith('$help'):
        await message.channel.send('Type $hey and then your message to talk to me!')
        
    if message.content.startswith('$hey'):
        content_str = message.content[4:]
        try:
            await message.channel.send(do_evaluate(encoder, decoder, searcher, vocabulary, content_str))
        except:
            await message.channel.send('I\'m unable to process your input. Sorry!')
        
client.run(TOKEN)
