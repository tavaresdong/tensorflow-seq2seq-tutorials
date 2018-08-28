import numpy as np
import tensorflow as tf
import helpers

from tensorflow.contrib.rnn import LSTMCell, GRUCell
from model_new import Seq2SeqModel, train_on_copy_task
import pandas as pd


tf.reset_default_graph()
tf.set_random_seed(1)

with tf.Session() as session:
    model = Seq2SeqModel(encoder_cell=LSTMCell(20),
                         decoder_cell=LSTMCell(20),
                         embedding_size=20,
                         vocab_size=10,
                         attention=False,
                         bidirectional=False,
                         beam_search=True,
                         debug=False)
    session.run(tf.global_variables_initializer())

    train_on_copy_task(session,
                       model,
                       length_from=3,
                       length_to=8,
                       vocab_lower=2,
                       vocab_upper=10,
                       batch_size=100,
                       max_batches=3000,
                       batches_in_epoch=1000,
                       verbose=True)


