import numpy as np
import tensorflow as tf
import helpers

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

tf.reset_default_graph()

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20

# Note that the decoer cell's hidden units doubles
decoder_hidden_units = encoder_hidden_units * 2

with tf.name_scope("input"):
    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name="encoder_inputs")
    encoder_inputs_length = tf.placeholder(shape=(None, ), dtype=tf.int32, name="encoder_inputs_length")
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name="decoder_targets")

with tf.name_scope("embedding"):
    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

# Build encoder graph
with tf.name_scope("encoder"):
    encoder_cell = LSTMCell(encoder_hidden_units)
    ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                        cell_bw=encoder_cell,
                                        inputs=encoder_inputs_embedded,
                                        sequence_length=encoder_inputs_length,
                                        dtype=tf.float32,
                                        time_major=True)
    # concat on axis 2
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    assert encoder_outputs.shape[2] == 2 * encoder_hidden_units

    # LSTM cell states contains c: hidden state, and h: output
    encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
    encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

    encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

# Build decoder graph
with tf.name_scope("decoder"):
    decoder_cell = LSTMCell(decoder_hidden_units)
    encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

    # we need to decide how long to run the decoder
    # 1. stop after a specified number of unrolling steps
    # 2. stop after model produced
    # here we unroll decoder for len(encoder_input) + 2 to allow model some room to make mistakes over 2 additional steps
    # <EOS> inputs + 2 more
    decoder_lengths = encoder_inputs_length + 3

    # specify the parameters used in global step and apply them at every step of the decoder
    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
    b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

    # we want to feed as decoder inputs tokens that it previously generated, instead
    # of the shifted sequence
    eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
    pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

    # decoder will operate on colunmn vectors of shape (batch_size, )
    eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
    pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)
    initial_inputs = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None
    return (initial_elements_finished,
            initial_inputs,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    def get_next_input():
        # project rnn output to shape [batch_size, vocab_size]
        output_logits = tf.add(tf.matmul(previous_output, W), b)

        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)
    finished = tf.reduce_all(elements_finished)
    input = tf.cond(finished, lambda:pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None: # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

with tf.name_scope("decoder"):
    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
    decoder_outputs = decoder_outputs_ta.stack()

    decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))

    # reshape to [max_time * batch_size, hidden_dim]
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
    decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

    # decoder_prediction is of shape [max_time, batch_size]
    decoder_prediction = tf.argmax(decoder_logits, 2)

with tf.name_scope("optimizer"):
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits
    )

    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)


batch_size = 100
batches = helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)
print("head of the batch:")
for seq in next(batches)[:10]:
    print(seq)

def next_feed():
    batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
    )

    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_
    }

max_batches = 3001
batches_in_epoch = 1000

with tf.Session() as sess:
    loss_track = []
    sess.run(tf.global_variables_initializer())
    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print(' minibatch loss {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('    sample {}'.format(i+1))
                    print('      input     > {}'.format(inp))
                    print('      predicted > {}'.format(pred))
                    if i >- 2:
                        break
                print()

    except KeyboardInterrupt:
        print('training interrupted')

