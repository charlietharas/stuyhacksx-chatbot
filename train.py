# -*- coding: utf-8 -*-

# Preliminary Initialiaztion

# imports
import pandas as pd
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

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--train_file", help="CSV file used for training.", required=True)
parser.add_argument("-u", "--user", help="User to mimic.", required=True)
parser.add_argument("-save_dir", "--save_filepath", help="Directory to save models in.", required=True)
parser.add_argument("-fa", "--feature_analysis", help="Include this flag to analyze some data features for potentially useful debugging.", action='store_true')
parser.add_argument("-w", "--words", help="Most used n words to print in feature analysis debugging.", default=100, type=int)
parser.add_argument("-te", "--tensordash_email", help="Email for streaming model progress to Tensordash app. Leave blank if not using.")
parser.add_argument("-tp", "--tensordash_password", help="Password for streaming model progress to Tensordash app. Leave blank if not using.")
parser.add_argument("-start_iter", "--start_iteration", help="Iteration to start at. Change if loading from checkpoint.", default=1, type=int)
parser.add_argument("-c", "--clip_value", help="Gradient clip. Default is 50.0.", default=50.0, type=float)
parser.add_argument("-tf", "--teacher_forcing_rate", help="Rate of teacher forcing. Default is 0.9.", default=0.9, type=float)
parser.add_argument("-tfr", "--teacher_forcing_ratio", help="Ratio of teacher forcing. Default is 1.0.", default=1.0, type=float)
parser.add_argument("-lr", "--learning_rate", help="Learning rate Alpha of model. Default is 0.0001.", default=0.0001, type=float)
parser.add_argument("-dlr", "--decoder_learning", help="Decoder learning rate multiplier. Default is 5.0.", default=5.0, type=float)
parser.add_argument("-i", "--num_iterations", help="Number of iterations or epochs of model. Default is 500.", default=500, type=int)
parser.add_argument("-pr", "--print_rate", help="Print output and send to Tensordash every n iterations. Default is 50.", default=50, type=int)
parser.add_argument("-sr", "--save_rate", help="Save model in .tar file every n iterations. Default is 100.", default=100, type=int)
parser.add_argument("-name", "--model_name", help="Customized name of model.", default='cb-model')
parser.add_argument("-attn_type", "--attention_model_type", help="Customized attention model type. Default is dot.", default='dot')
parser.add_argument("-hidden", "--hidden_layer_size", help="Size of hidden layer. Default is 512.", default=512, type=int)
parser.add_argument("-e_layers", "--encoder_layers", help="Number of layers in the encoder. Default is 2.", default=2, type=int)
parser.add_argument("-d_layers", "--decoder_layers", help="Number of layers in the decoder. Default is 2.", default=2, type=int)
parser.add_argument("-drop", "--dropout_rate", help="Rate of dropout of the model. Default is 0.1.", default=0.1, type=float)
parser.add_argument("-batch_size", "--batch_size", help="Size of batches in the model. Default is 64.", default=64, type=int)

args = parser.parse_args()
filename = args.train_file
sel_user = args.user
feature_analysis = args.feature_analysis
top_words = args.words
tensordash_user = args.tensordash_email
tensordash_pass = args.tensordash_password

# model parameters
clip = args.clip_value
teacher_forcing = args.teacher_forcing_rate
alpha = args.learning_rate
decoder_learning = args.decoder_learning
n_iter = args.num_iterations 
print_rate = args.print_rate
save_rate = args.save_rate
teacher_forcing_ratio = args.teacher_forcing_ratio
model_name = args.model_name
attn_model = args.attention_model_type
hidden_size = args.hidden_layer_size
encoder_n_layers = args.encoder_layers
decoder_n_layers = args.decoder_layers
dropout = args.dropout_rate
batch_size = args.batch_size
train_loss = []
save_directory = args.save_filepath

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

"""
# "Count"-Type Analysis (OPTIONAL)

1.   Performs a total message count and analyzes n (eg 100, specify in flag) most used words.
2.   Counts most prominent authors sending messages prior to those of sel_user (eg most common people conversed with/after).
3.   Counts participation in all unique conversations (conversations defined as exchanges where time between messages <30min).
4.   Counts distinct groups of people conversed with.
5.   Counts the amount of words from each user (basic).
"""

if feature_analysis:
    print("ANALYZING FEATURES OF YOUR DATA. This will not affect training or application performance, but may help with debugging.")
    print("Remove the --feature-analysis flag to move straight to training.")
    
    usr_message_count = 0
    total_word_count = 0
    words = []
    word_count = []
    
    for i in range (content.size):
        if str(authors.get(i)) == sel_user:
            usr_message_count += 1;
            word_in_row = str(content.get(i)).split()
            for j in word_in_row:
                words.append(j)
                total_word_count += 1
    
    wordset = set(words)
    print("Total words from " + sel_user + ": " + str(total_word_count))
    print("Total messages from " + sel_user + ": " + str(usr_message_count))
    
    for i in wordset:
        word_count.append([i, words.count(i)])
    excluded_words = set()
    most_used_words = []
    for x in range(top_words): # bad sorting algorithm
        max_i = 0
        max_i_word = ""
        for i in word_count:
            if len(i[0]) > 0 and i[0] not in excluded_words:
                if i[1] > max_i:
                    max_i_word = i[0]
                    max_i = i[1]
        excluded_words.add(max_i_word)
        most_used_words.append([max_i_word, max_i])
    
    print("Most used " + str(top_words) + " words: " + str(most_used_words))
    print("Total (non-distinct) words: " + str(total_word_count) + " at average of " + str(total_word_count/usr_message_count) + " words per message.")
    
    authorlist = []
    author_count = []
    
    for i in range (authors.size):
        if str(authors.get(i)) == sel_user:
            authorlist.append(authors.get(i-1))
    
    authorset = set(authorlist)
    
    for i in authorset:
        author_count.append([i, authorlist.count(i)])
    
    print("Amount of times each author sent a message before one from " + sel_user + ": " + str(author_count))
    
    convset = set()
    
    for i in range(conv_id.size):
        if str(authors.get(i)) == sel_user:
            convset.add(conv_id.get(i))
    
    print("Got " + str(int(conv_id.get(conv_id.size-1))) + " distinct conversations, " + sel_user+ ", participation in " + str(len(convset)) + " of them at rate " + str(len(convset)/int(conv_id.get(conv_id.size-1))))
    
    authors_permutations = []
    convID = -1
    
    for i in range(conv_id.size):
        if conv_id.get(i) in convset:
            if conv_id.get(i) != convID:
                authors_permutations.append(set())
            else:
                authors_permutations[len(authors_permutations)-1].add(authors.get(i))
            convID = conv_id.get(i)
    
    # creates set of permutations (list)
    authors_permutations_included = []
    for i in authors_permutations:
        if i in authors_permutations_included:
            continue
        else:
            authors_permutations_included.append(i)
    
    authors_permutations_count = []
    for i in authors_permutations_included:
        authors_permutations_count.append([i, authors_permutations.count(i)])
    
    # get unsorted list of distinct conversational groups
    print("Each distinct group of participants in conversations, with frequency amount:")
    print(authors_permutations_count)
    
    # bug noticed: prints [set(), n], but.. whatever

"""# Sequence to Sequence

Vectorizing the dictionary of distinct words as a Vocabulary object, grabbing specialized conversation pairs, and training a model to respond. This is the core of the project, inspired by and heavily relying on content from https://medium.com/swlh/end-to-end-chatbot-using-sequence-to-sequence-architecture-e24d137f9c78.
"""

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
            if token == value:
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

# loss function
def loss_func(inpt, target, mask):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inpt, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.item()

# training functions
device = torch.device("cpu") # program currently only supports CPU training
def train(input_variable, lengths, target_variable, mask, max_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor([[SRT for i in range (batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, n_total = loss_func(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, n_total = loss_func(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    
    loss.backward()

    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def train_iterations(model_name, vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, n_iterations, batch_size, print_rate, save_rate, clip):
    training_batches = [batch_to_data(vocabulary, [random.choice(pairs) for i in range(batch_size)]) for ii in range(n_iterations)]

    start_iteration = args.start_iteration
    print_loss = 0
    
    for iteration in range(start_iteration, n_iterations + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # tensordash
        if (tensordash_user is not None and tensordash_pass is not None):
            histories.sendLoss(loss = loss, epoch = iteration, total_epochs = n_iterations+1)

        if iteration % print_rate == 0:
            print_loss_avg = print_loss / print_rate
            train_loss.append(print_loss_avg)
            print("Iteration {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration/n_iterations*100, print_loss_avg))
            print_loss = 0

        if iteration % save_rate == 0:
            directory = os.path.join(save_directory, sel_user, model_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': vocabulary.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

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

# training the model
embedding = nn.Embedding(vocabulary.num_words, hidden_size)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocabulary.num_words, decoder_n_layers, dropout)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=alpha)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=alpha * decoder_learning)

encoder.train()
decoder.train()

# the training function
train_iterations(model_name, vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, n_iter, batch_size, print_rate, save_rate, clip)
