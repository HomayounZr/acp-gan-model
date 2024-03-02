import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import time

print(tf.__version__)

from tensorflow.keras.utils import to_categorical as labelEncoding 

T = 15 # terminus_length

X1 = np.load('bpf-740.npy')
X2 = np.load('bits-740.npy')
X3 = np.load('blosum-740.npy')


X1 = X1[:,0:T,:]
X2 = X2[:,0:T,:]
X3 = X3[:,0:T,:]


Y  = [1 for _ in range(376)]
Y += [0 for _ in range(364)]

Y = labelEncoding(Y, dtype=int)

print(Y.shape)

print(X1.shape)
print(X1[0].shape)
print(X1[0])
print(X2.shape)
print(X2[0].shape)
print(X2[0])
print(X3.shape)
print(X3[0].shape)
print(X3[0])

# Deep Neural Networks:
import tensorflow as tf; print('We\'re using TF-{}.'.format(tf.__version__))
# import keras; print('We\'re using Keras-{}.'.format(keras.__version__))
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten, BatchNormalization,
                                     Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,
                                     LSTM, GRU, Embedding, Bidirectional, Concatenate)
from tensorflow.keras.regularizers import (l1, l2, l1_l2)
from tensorflow.keras.optimizers import (RMSprop, Adam, SGD)
from tensorflow.keras.models import (Sequential, Model)

# Core:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
import matplotlib.patches as patches

# Performance:
from sklearn.metrics import (confusion_matrix, classification_report, matthews_corrcoef, precision_score, roc_curve, auc)
from sklearn.model_selection import (StratifiedKFold, KFold, train_test_split)

#Utilities:
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical as labelEncoding   # Usages: Y = labelEncoding(Y, dtype=int)
from tensorflow.keras.utils import plot_model

def build_discriminator():
    ### Head-1:
    input1 = Input(shape=X1[0].shape)

    x = Conv1D(filters=10, kernel_size=4, padding='same', activation='relu', kernel_regularizer=l2(l=0.01))(input1)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.80)(x)

    x = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(l=0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.70)(x)

    head1 = Flatten()(x)


    ### Head-2:
    # input2 = Input(shape=X2[0].shape)

    # x = Conv1D(filters=10, kernel_size=4, padding='same', activation='relu', kernel_regularizer=l2(l=0.01))(input2)
    # x = BatchNormalization()(x)
    # x = Dropout(rate=0.70)(x)

    # x = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(l=0.01))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(rate=0.70)(x)

    # head2 = Flatten()(x)


    ### Head-3:
    input3 = Input(shape=X3[0].shape)

    x = Conv1D(filters=10, kernel_size=4, padding='same', activation='relu',)(input3)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.70)(x)

    x = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(l=0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.70)(x)

    head3 = Flatten()(x)


    # merge
    merge = Concatenate()([head1, head3])

    output = Dense(units=8, activation='relu', kernel_regularizer=l2(l=0.01))(merge)
    output = BatchNormalization()(output)
    output = Dropout(rate=0.70)(output)

    output = Dense(units=2, activation='softmax')(output)

    return Model(inputs=[input1, input3], outputs=[output])

def build_generator(latent_dim, output_shape):
    model = Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(output_shape[0] * output_shape[1], activation='sigmoid'),
        layers.Reshape(output_shape),
        layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=-1), depth=output_shape[1])),
        layers.Lambda(lambda x: tf.cast(x, dtype=tf.int32))
    ])
    return model

# Define the dimensions
latent_dim = 100  # Dimensionality of the latent space
output_shape = (25, 20)  # Desired output shape of the matrix

generator = build_generator(latent_dim, output_shape)

noise = tf.random.normal([1, latent_dim])

generated_matrix = generator(noise, training=False)

print(generated_matrix)

from Bio.Seq import Seq
from Bio.SeqUtils import ProtParam
from Bio.Align import substitution_matrices
from Bio.SubsMat import MatrixInfo


# Function to convert binary profile feature to amino acid sequence
def bpf_to_sequence(binary_profile_feature):
    amino_acids = "ARNDCQEGHILKMFPSTWYVBZX"
    sequence = ""
    for row in binary_profile_feature:
        index = row.argmax()
        sequence += amino_acids[index]
    return sequence

def seq_to_blosum(sequence):
    blosum62 = MatrixInfo.blosum62
    print(blosum62)
    sequence = sequence.upper()
    length = len(sequence)
    blosum_matrix = []
    for aa1 in sequence:
        row = []
        for aa2 in sequence:
            if (aa1, aa2) in blosum62:
                row.append(blosum62[(aa1, aa2)])
            else:
                row.append(blosum62[(aa2, aa1)])
        blosum_matrix.append(row)
    return blosum_matrix

fake_bpf = generated_matrix[0].numpy()

print("Binary Profile Feature Matrix:")
print(fake_bpf.tolist())
print()

# Convert binary profile feature to amino acid sequence
sequence = bpf_to_sequence(generated_matrix[0].numpy())
print("Amino Acid Sequence:")
print(sequence)
print()

# blosum62 = substitution_matrices.load("BLOSUM62")

print("BLOSUM62 Matrix:")
fake_blosum = seq_to_blosum(sequence)
print(fake_blosum)

# Calculate physiochemical properties
# protein_seq = Seq(sequence)
# protein_analyzer = ProtParam.ProteinAnalysis(str(protein_seq))
# physiochemical_matrix = protein_analyzer.protein_scale(window=5)

# Display physiochemical properties matrix
print("Physiochemical Properties Matrix:")
# print(physiochemical_matrix)

def convert_input(generated_seq):
    fake_bpf = np.array(generated_seq)
    seq = bpf_to_sequence(fake_bpf)
    return np.array(fake_bpf), np.array(seq_to_blosum(seq))

discriminator = build_discriminator()
discriminator.load_weights('./acp_mhcnn_weights.h5')

prediction = discriminator.predict([X1[:5,:,:], X3[:5,:,:]])
print(prediction)

noise = tf.random.normal([5, 100])
fake_seqs = generator(noise, training=False)
print(fake_seqs)

print(fake_seqs)
inputs_bpf = []
inputs_blosum = []
for seq in fake_seqs:
    bpf, blosum = convert_input(seq)
    inputs_bpf.append(bpf[:15])
    inputs_blosum.append(blosum[:15,:20])

inputs_bpf = np.array(inputs_bpf)
inputs_blosum = np.array(inputs_blosum)

# print(X3[:5,:,:].shape)
# print(X3[0])
# print('---------------')
# print(inputs_blosum.shape)
# print(inputs_blosum[0])

prediction = discriminator.predict([inputs_bpf, inputs_blosum])
print(prediction)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

import os

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
BATCH_SIZE = 32

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

fake_seqs = tf.Variable(initial_value=tf.zeros((32, 25, 20), dtype=tf.int32), trainable=False, dtype=tf.int32)

# @tf.function
def train_step():
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    fake_seqs_sym = generator(noise, training=True)
    # print(fake_seqs_sym)
    # print(fake_seqs)
    
    inputs_bpf = []
    inputs_blosum = []
    for seq in fake_seqs_sym:
        bpf, blosum = convert_input(seq)
        inputs_bpf.append(bpf[:15])
        inputs_blosum.append(blosum[:15,:20])

    inputs_bpf = np.array(inputs_bpf)
    inputs_blosum = np.array(inputs_blosum)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:


      # real_output = discriminator(images, training=False)
      fake_output = discriminator.predict([inputs_bpf, inputs_blosum])
      print(fake_output)
    return

    #   gen_loss = generator_loss(fake_output)
    #   # disc_loss = discriminator_loss(real_output, fake_output)

    #   gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    #   # gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    #   generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    #   # discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(epochs):
  for epoch in range(epochs):
    start = time.time()

    train_step()
    break
  
train(EPOCHS)