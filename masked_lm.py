import sys, os
import tensorflow as tf
from bert_modified import modeling
import numpy as np
from bert_modified import tokenization
import tensorflow.contrib.keras as kr
import json
import warnings
import os
import pickle

####################################################################################################

__author__ = 'Yuzhong Hong <hongyuzhong@qiyi.com>'
__date__, __version__ = '02/26/2019', '0.1'  # Module Creation.
__date__, __version__ = '04/04/2019', '0.2'  # Add on-demand n-gram masked language model


__description__ = 'Masked language model'

__future_work__ = '1. improve computational efficiency by changing scalar computation to matrix computation'

####################################################################################################


warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# BIGRAMS = pickle.load(open('bigram_dict_simplified.sav', 'rb'))


class Config(object):
    max_seq_length = 16
    vocab_file = "model/pre-trained/vocab.txt"
    bert_config_file = "model/pre-trained/bert_config.json"
    init_checkpoint = "model/pre-trained/bert_model.ckpt"
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    topn = 5
    bigrams = None  # pickle.load(open('bigram_dict_simplified.sav', 'rb'))


class Model(object):
    def __init__(self, config):
        self.config = config

        # placeholders
        self.input_ids = tf.placeholder(tf.int32, [None, self.config.max_seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.config.max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, config.max_seq_length], name='segment_ids')
        self.masked_lm_positions = tf.placeholder(tf.int32, [None, self.config.max_seq_length - 2],
                                                  name='masked_lm_positions')
        self.masked_lm_ids = tf.placeholder(tf.int32, [None, self.config.max_seq_length - 2],
                                            name='masked_lm_ids')
        self.masked_lm_weights = tf.placeholder(tf.float32, [None, self.config.max_seq_length - 2],
                                                name='masked_lm_weights')

        is_training = False

        # create model
        masked_lm_loss, masked_lm_example_loss, self.masked_lm_log_probs, self.probs = self.create_model(
            self.input_ids,
            self.input_mask,
            self.segment_ids,
            self.masked_lm_positions,
            self.masked_lm_ids,
            self.masked_lm_weights,
            is_training,
            config.bert_config)

        # prediction
        self.masked_lm_predictions = tf.argmax(self.masked_lm_log_probs, axis=-1, output_type=tf.int32)
        self.top_n_predictions = tf.nn.top_k(self.probs, k=config.topn, sorted=True, name="topn")

    def predict(self, batch, sess):
        """
        for predicting
        """

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = batch

        feed_dict = {
            self.input_ids: input_ids,
            self.input_mask: input_mask,
            self.segment_ids: segment_ids,
            self.masked_lm_positions: masked_lm_positions,
            self.masked_lm_ids: masked_lm_ids,
            self.masked_lm_weights: masked_lm_weights
        }

        masked_lm_predictions, masked_lm_log_probs = sess.run(
            [self.masked_lm_predictions, self.masked_lm_log_probs], feed_dict)

        return masked_lm_predictions

    def topn_predict(self, batch, sess):
        """
        for predicting topn results
        """

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = batch

        feed_dict = {
            self.input_ids: input_ids,
            self.input_mask: input_mask,
            self.segment_ids: segment_ids,
            self.masked_lm_positions: masked_lm_positions,
            self.masked_lm_ids: masked_lm_ids,
            self.masked_lm_weights: masked_lm_weights
        }

        top_n_predictions = sess.run(self.top_n_predictions, feed_dict)
        topn_probs, topn_predictions = top_n_predictions

        return np.array(topn_probs, dtype=float), topn_predictions

    def create_model(self,
                     input_ids,
                     input_mask,
                     segment_ids,
                     masked_lm_positions,
                     masked_lm_ids,
                     masked_lm_weights,
                     is_training,
                     bert_config):
        """Create Masked Language Model"""

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs, probs = self.get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids, masked_lm_weights)

        return masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs, probs

    @classmethod
    def get_masked_lm_output(cls, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights):
        """Get loss and log probs for the masked LM."""
        input_tensor = cls.gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

        return loss, per_example_loss, log_probs, probs

    @staticmethod
    def gather_indexes(sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor


class MaskedLM(object):
    def __init__(self, config):
        self.config = config

        # create session
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)
        session_conf.gpu_options.allow_growth = True
        self.session = tf.Session(config=session_conf)

        # load model
        self.model = self.load_model(config)
        self.session.run(tf.global_variables_initializer())

        self.processor = Processor(config.vocab_file, config.max_seq_length)

    @staticmethod
    def load_model(config):

        model = Model(config)

        tvars = tf.trainable_variables()

        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)

        tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)

        return model


    def find_topn_candidates(self, sentences, batch_size=1):
        """
        Args
        -----------------------------
        sentences: a list of sentences, e.g., ['the man went to the store.', 'he bought a gallon of milk.']
        batch_size: default=1

        Return
        -----------------------------
        candidates for each token in the sentences, e.g., [[[('the', 0.88), ('a', 0.65)], ...], [...]]

        """
        data = Data(sentences, self.processor)
        stream_res = []
        stream_probs = []
        lengths = []
        while True:
            batch = data.next_predict_batch(batch_size)
            if batch is not None:
                _, id_mask_batch, _, _, _, _ = batch
                topn_probs, topn_predictions = self.model.topn_predict(batch, self.session)
                lengths.extend(list(np.sum(id_mask_batch, axis=-1)))
                stream_res.extend(topn_predictions)
                stream_probs.extend(topn_probs)
            else:
                break

        res = []
        pos = 0
        length_id = 0

        while pos < len(stream_res):
            sen = []
            for i in range(self.config.max_seq_length - 2):
                if i < lengths[length_id] - 2:  # to account for [CLS] and [SEP]
                    token_candidates = []
                    for token_idx, prob in zip(stream_res[pos], stream_probs[pos]):
                        token_candidates.append((self.processor.idx_to_word[token_idx], prob))
                    sen.append(token_candidates)
                pos += 1
            length_id += 1
            res.append(sen)
        

        return res


class Data(object):
    """
    Load data.

    """

    def __init__(self, data, processor):

        self.data = data
        self.pos = 0  # records the iterating progress for df
        self.processor = processor

    def next_predict_batch(self, batch_size):
        """
        Produce the next batch for predicting.

        Args
        ----------------
        batch_size: batch_size for predicting

        Returns
        ----------------
        features_padded_batch, tags_padded_batch, length_batch
        or
        None if the data is exhausted
        """
        print(f'processed {self.pos} entries...')
        if self.pos >= len(self.data):
            self.pos = 0  # get ready for the next round of prediction

            return None

        else:
            batch = self.data[self.pos: self.pos + batch_size]
            self.pos += batch_size

            input_ids_batch, \
            input_mask_batch, \
            segment_ids_batch, \
            masked_lm_positions_batch, \
            masked_lm_ids_batch, \
            masked_lm_weights_batch = self.parse(batch)

            input_ids_batch = kr.preprocessing.sequence.pad_sequences(input_ids_batch,
                                                                      self.processor.max_seq_length,
                                                                      padding='post')
            input_mask_batch = kr.preprocessing.sequence.pad_sequences(input_mask_batch,
                                                                       self.processor.max_seq_length,
                                                                       padding='post')
            segment_ids_batch = kr.preprocessing.sequence.pad_sequences(segment_ids_batch,
                                                                        self.processor.max_seq_length,
                                                                        padding='post')

            masked_lm_positions_batch = kr.preprocessing.sequence.pad_sequences(masked_lm_positions_batch,
                                                                                self.processor.max_seq_length - 2,
                                                                                padding='post')
            masked_lm_ids_batch = kr.preprocessing.sequence.pad_sequences(masked_lm_ids_batch,
                                                                          self.processor.max_seq_length - 2,
                                                                          padding='post')
            masked_lm_weights_batch = kr.preprocessing.sequence.pad_sequences(masked_lm_weights_batch,
                                                                              self.processor.max_seq_length - 2,
                                                                              padding='post')

            return input_ids_batch, input_mask_batch, segment_ids_batch, masked_lm_positions_batch, masked_lm_ids_batch, masked_lm_weights_batch

    def parse(self, batch):
        input_ids_batch, \
        input_mask_batch, \
        segment_ids_batch, \
        masked_lm_positions_batch, \
        masked_lm_ids_batch, \
        masked_lm_weights_batch = [], [], [], [], [], []
        for sentence in batch:

            input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = \
                self.processor.create_single_instance(sentence)
            input_ids_batch.append(input_ids)
            input_mask_batch.append(input_mask)
            segment_ids_batch.append(segment_ids)
            masked_lm_positions_batch.append(masked_lm_positions)
            masked_lm_ids_batch.append(masked_lm_ids)
            masked_lm_weights_batch.append(masked_lm_weights)

        return input_ids_batch, input_mask_batch, segment_ids_batch, masked_lm_positions_batch, masked_lm_ids_batch, masked_lm_weights_batch


class Processor(object):
    def __init__(self, vocab_file, max_seq_length):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
        self.idx_to_word = self.inverse_vocab(self.tokenizer.vocab)
        self.max_seq_length = max_seq_length

    @staticmethod
    def inverse_vocab(vocab):
        idx_to_word = {}
        for word in vocab:
            idx_to_word[vocab[word]] = word
        return idx_to_word

    def create_single_instance(self, sentence):
        # tokenization
        tokens_raw = self.tokenizer.tokenize(tokenization.convert_to_unicode(sentence))

        # add [CLS] and [SEP]
        assert len(sentence) <= self.max_seq_length - 2
        tokens = ["[CLS]"] + tokens_raw + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # produce pseudo ground truth, since the truth is unknown when it comes to spelling checking.
        input_tokens, masked_lm_positions, masked_lm_labels = self.create_pseudo_ground_truth(tokens)

        # convert to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(segment_ids)

        masked_lm_positions = list(masked_lm_positions)
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        # print(input_tokens)

        return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights

    @staticmethod
    def create_pseudo_ground_truth(tokens):
        input_tokens = list(tokens)
        masked_lm_positions = []
        masked_lm_labels = []

        for index, token in enumerate(tokens):

            if token == "[CLS]" or token == "[SEP]":
                continue

            masked_token = tokens[index]  # keep the original token

            input_tokens[index] = masked_token
            masked_lm_positions.append(index)
            masked_lm_labels.append(tokens[index])

        return input_tokens, masked_lm_positions, masked_lm_labels



# def test_masked_lm():
#     config = Config()
#     lm = MaskedLM(config)
#     res = lm.find_topn_candidates(
#          ['。国际电台苦名丰持人。'], 2)
#     for sen in res:
#         print(sen)


# if __name__ == '__main__':
#     test_masked_lm()