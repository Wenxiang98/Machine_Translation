import time
import MeCab
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from IPython.display import display

nltk.download('punkt')

file_path = 'jpn.txt'
cleaned_file_path = 'cleaned_jpn_eng.txt'

def clean_text(file_path, cleaned_file_path):
    df = pd.read_table(file_path, names=['target','source','noise'])
    df = df.iloc[:,:2]
    df = df.reindex(columns=['source','target'])
    df.insert(1,'seperator',['\t' for i in range(df.shape[0])])
    np_array = df.to_numpy()
    np.savetxt(cleaned_file_path,np_array,fmt="%s", encoding = 'utf-8')
    print("[Created]{}".format(cleaned_file_path))

clean_text(file_path,cleaned_file_path)

##runtimeError occurs here
mecab = MeCab.Tagger('-Owakati')

with open('cleaned_jpn_eng.txt', mode = 'rt', encoding='utf-8') as f:
    lines = f.read().split('\n')

jpn_dataset = list()
eng_dataset = list()
lower_chars = [chr(i) for i in range(97,123)]

def analyze_jpn(sentence):
    ret = mecab.parse(sentence).split(' ')
    ret.remove('\n')
    return ret
    
def analyze_eng(sentence):
    return word_tokenize(sentence.lower())

for i in range(len(lines)):
    is_lower = False
    line = lines[i].split("\t")
    for char in line[0].strip():
        if char.lower() in lower_chars:
            is_lower = True
        
        if not is_lower:
            jpn_line = analyze_jpn(line[0].strip())
            eng_line = analyze_eng(line[-1].strip())
            jpn_dataset.append(jpn_line)
            eng_dataset.append(eng_line)

jpn_dataset = jpn_dataset[:-1]
eng_dataset = eng_dataset[:-1]
print(len(jpn_dataset))
print(len(eng_dataset))

for index, jpn_line in enumerate(jpn_dataset):
    jpn_dataset[index] = ['<bos>'] + jpn_line + ['<eos>']
for index, eng_line in enumerate(eng_dataset):
    eng_dataset[index] = ['<bos>'] + eng_line + ['<eos>']

def tokenize(x):
    x_tk = tf.keras.preprocessing.text.Tokenizer(filters = '', char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk, x_tk.texts_to_sequences(x)

source_sentence_tokenizer, source_tensor = tokenize(np.array(jpn_dataset))
target_sentence_tokenizer, target_tensor = tokenize(np.array(eng_dataset))
source_tensor = tf.keras.preprocessing.sequence.pad_sequences(source_tensor,padding = 'post')
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,padding = 'post')

source_vocab_size = len(source_sentence_tokenizer.word_index) + 1
target_vocab_size = len(target_sentence_tokenizer.word_index) + 1
max_length_inp = source_tensor.shape[1]
max_length_targ = target_tensor.shape[1]

print("[Source] Japanese vocabulary size: ", source_vocab_size)
print("[Target] English vocabulary size: ", target_vocab_size)
print("[Source] Max Japanese sentence length: ", max_length_inp)
print("[Target] Max English sentence length: ", max_length_targ)

input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test =\
    train_test_split(source_tensor, target_tensor, test_size = 0.2)

print(len(input_tensor_test),len(target_tensor_train), len(input_tensor_test), len(target_tensor_test))

def convert_Id_to_World(lang, tensor):
    ids = []
    words = []
    for t in tensor:
        if t!=0:
            ids.append(t)
            words.append(lang.index_word[t])
        return ids, words

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(source_sentence_tokenizer.word_index)+1
vocab_tar_size = len(target_sentence_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train,target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)

###Encoder class
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

###Bahdanau Attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, query, values):
        query_with_time_axis = tf.expand_dims(query,1)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

##Decoder class
class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim,  dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

        #used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def __call__(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output,(-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

class Seq2Seq_Attention(tf.keras.Model):
    def __init__(self, encoder_dict, decoder_dict):
        super(Seq2Seq_Attention, self).__init__()
        self.encoder = Encoder(**encoder_dict)
        self.decoder = Decoder(**decoder_dict)

    def __call__(self, inp, targ, enc_hidden):
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_sentence_tokenizer.word_index['<bos>']] * BATCH_SIZE, 1)

        return self.Teacher_Forcing(targ, dec_input, dec_hidden, enc_output)

    def Teacher_Forcing(self, targ, dec_input, dec_hidden, enc_output):

        loss = 0
    
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            loss +=loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t],1)
        
        return loss 

encoder_dict = {'vocab_size': vocab_inp_size,
                'embedding_dim': embedding_dim,
                'enc_units': units,
                'batch_sz': BATCH_SIZE
                }

decoder_dict = {'vocab_size': vocab_tar_size,
                'embedding_dim': embedding_dim,
                'dec_units': units,
                'batch_sz': BATCH_SIZE}

model = Seq2Seq_Attention(encoder_dict, decoder_dict)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype = loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

checkpoint_path = "./checkpoint/train"

ckpt = tf.train.Checkpoint(model=model,
                            optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
else:
    print("Initializing from scratch.")

@tf.function
def train_step(inp, targ, enc_hidden):

    with tf.GradientTape() as tape:
        loss = model(inp, targ, enc_hidden)

    batch_loss = (loss / int(targ.shape[1]))

    variables = model.encoder.trainable_variables + model.decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

EPOCHS = 15

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = model.encoder.initialize_hidden_state()
    total_loss = 0

    for(batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 ==0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))


    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))

    print('Time taken for epoch {} sec\n'.format.time() . start)

def predict(sentence):

    sentence = analyze_jpn(sentence)

    inputs = [source_sentence_tokenizer.word_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen = max_length_inp,
                                                            padding = 'post')

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeroes((1,units))]
    enc_out, enc_hidden = model.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_sentence_tokenizer.word_index['<bos>']],0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = model.decoder(dec_input,
                                                                    dec_hidden,
                                                                    enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_sentence_tokenizer.index_word[predicted_id] + ' '

        if target_sentence_tokenizer.index_word[predicted_id] == '<eos>':
            return result, sentence

def translate(sentence):
    result, sentence = predict(sentence)
    print('Input: {}'.format(''.join(sentence)))
    print('Translatation: {}'.format(result[:-6]))

def evaluate(references, candidates):
    cumulative_1gram = list()
    cumulative_2grams = list()
    cumulative_3grams = list()
    cumulative_4grams = list()

    for reference, candidates in zip(references, candidates):
        cumulative_1gram.append(sentence_bleu([reference],candidate, weights=(1,0,0,0)))
        cumulative_2grams.append(sentence_bleu([reference],candidate, weights=(0.5,0.5,0,0)))
        cumulative_3grams.append(sentence_bleu([reference],candidate, weights=(0.33,0.33,0.33,0)))
        cumulative_4grams.append(sentence_bleu([reference],candidate, weights=(0.25,0.25,0.25,0.25)))
        
    return [cumulative_1gram, cumulative_2grams, cumulative_3grams, cumulative_4grams]

references = list()
candidates = list()

for index in range(5):
    _, reference = convert_Id_to_World(target_sentence_tokenizer,target_tensor_test[index])
    _, candidate = convert_Id_to_World(source_sentence_tokenizer, input_tensor_test[index])
    candidate, _ = predict(''.join(candidate[1:-1]))
    references.append(''.join(reference[1:-1]))
    candidates.append(candidate[:-6])

sources = evaluate(references, candidates)

top_df = {
    'Original': references,
    'Predicted': candidates,
    'BLEU_1': scores[0],
    'BLEU_2': scores[1],
    'BLEU_3': scores[2],
    'BLEU_4': scores[3]
}

table_off = pd.DataFrame(to_df)
table_off.columns = ['Original','Predicted','BLEU_1','BLEU_2','BLEU_3','BLEU_4']

display(table_df)