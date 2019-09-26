import abc
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import codecs

VOCAB_SIZE_THRESHOLD = 50000

tf.enable_eager_execution()


class EmbeddingInterface(abc.ABC):
    @abc.abstractmethod
    def encoder_embedding(self):
        """Create embedding for encoder."""
        raise NotImplementedError()

    @abc.abstractmethod
    def decoder_embedding(self):
        """Create embedding for decoder."""
        raise NotImplementedError()

    @abc.abstractmethod
    def encoder_embedding_input(self, inputs):
        """Create encoder embedding input.

        Args:
          inputs: A tf.string tensor

        Returns:
          embedding presentation of inputs
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decoder_embedding_input(self, inputs):
        """Create decoder embedding input.

        Args:
          inputs: A tf.string tensor

        Returns:
          embedding presentation of inputs
        """
        raise NotImplementedError()


class AbstractEmbedding(EmbeddingInterface):
    def __init__(self, params, embedding_scope="embedding"):
        self.embedding_scope = embedding_scope
        self.params = params
        self._encoder_embedding = None
        self._decoder_embedding = None
        self.unk = "<blank>"
        self.unk_id = 0
        self.src_vocab_file = params["vocabs_features_file"]
        self.tgt_vocab_file = params["vocabs_labels_file"]

        self.src_str2idx_table = lookup_ops.index_table_from_file(
            self.src_vocab_file, default_value=self.unk_id)
        self.src_idx2str_table = lookup_ops.index_to_string_table_from_file(
            self.src_vocab_file, default_value=self.unk)
        self.tgt_str2idx_table = lookup_ops.index_table_from_file(
            self.tgt_vocab_file, default_value=self.unk_id)
        self.tgt_idx2str_table = lookup_ops.index_to_string_table_from_file(
            self.tgt_vocab_file, default_value=self.unk)
        print(self.src_str2idx_table)

    def encoder_embedding(self):
        return self._encoder_embedding

    def decoder_embedding(self):
        return self._decoder_embedding

    def encoder_embedding_input(self, inputs):
        '''
        source sequence embedding
        embedding str2int
        :param src:
        :return:
        '''
        params = self.params
        inputs_ids = self.src_str2idx_table.lookup(inputs)
        print("inputs_ids:\n", inputs_ids)
        vocab_size = params["src_vocab_size"]
        embedding_size = params["src_embedding_size"]
        with tf.variable_scope(self.embedding_scope, dtype=tf.float32, reuse=tf.AUTO_REUSE) as scope:
            self._encoder_embedding = self._create_or_load_embedding(name="encoder_embed", vocab_size=vocab_size,
                                                                     embedding_size=embedding_size,
                                                                     embedding_file=None, vocab_file=None)
            embedding_inputs = tf.nn.embedding_lookup(self._encoder_embedding, inputs_ids, name="embed_inputs")
        return embedding_inputs,inputs_ids

    def decoder_embedding_input(self, inputs):
        '''
        target sequence embedding
        :param inputs:
        :param params:
        :return:
        '''
        params = self.params
        target_inputs = self.tgt_str2idx_table.lookup(inputs)
        vocab_size = params["tgt_vocab_size"]
        embedding_size = params["tgt_embedding_size"]
        self._decoder_embedding = self._create_or_load_embedding(name="decoder_embed", vocab_size=vocab_size,
                                                                 embedding_size=embedding_size,
                                                                 embedding_file=None, vocab_file=None)
        embedding_inputs = tf.nn.embedding_lookup(self._decoder_embedding, target_inputs, name="embed_inputs")
        return embedding_inputs

    def _create_or_load_embedding(self, name, vocab_size, embedding_size, embedding_file, vocab_file=None):
        '''
        if embedding file exists.load embedding from embedding file,
            else create embedding variable
        :param vocab_size:
        :param embedding_size:
        :param embedding_file:
        :return:
        '''
        if embedding_file and vocab_file:
            embedding = self._load_pretrained_embedding(embedding_file, vocab_file)
        else:
            with tf.device(self._create_embedding_device(vocab_size)):
                embedding = tf.get_variable(name=self.embedding_scope, dtype=tf.float32,
                                            shape=[vocab_size, embedding_size])
        return embedding

    def _load_pretrained_embedding(self, embedding_file, vocab_file):
        raise NotImplementedError()

    @staticmethod
    def _create_embedding_device(vocab_size):
        if vocab_size > VOCAB_SIZE_THRESHOLD:
            return "/cpu:0"
        else:
            return "/cpu:0"


class NMTEmbedding(AbstractEmbedding):

    def _load_pretrained_embedding(self, embedding_file, vocab_file,
                                   num_trainable_tokens=3,
                                   dtype=tf.float32,
                                   scope="pretrained_embedding"):
        """
        处理预训练的词嵌入
        :param embedding_file: 预训练的词嵌入文件
        :param vocab_file: 收集的词典文件
        :return: embedding matrix
        """
        vocabs, vocabs_len = self._load_vocabs_file(vocab_file=vocab_file)
        embedding_dict, embedding_size = self._load_embedding_file(embedding_file)
        vocabs_vec = []
        for vocab in vocabs:
            if vocab in embedding_dict.keys:
                vocabs_vec.append(embedding_dict.get(vocab))
            else:
                vocab_vec = [0.0] * embedding_size
                vocabs_vec.append(vocab_vec)
        matrix = tf.constant(vocabs_vec, dtype=tf.float32)

        embedding_matrix_const = tf.slice(matrix, [num_trainable_tokens, 0], [-1, -1])

        with tf.variable_scope(self.embedding_scope, reuse=tf.AUTO_REUSE, dtype=tf.float32) as scope:
            # 将<blank>,<s>,</s> 单独处理，重新训练这三个值的vec
            embedding_matrix_variable = tf.get_variable(
                "embedding_matrix_variable",
                [num_trainable_tokens, embedding_size])
        return tf.concat([embedding_matrix_variable, embedding_matrix_const], 0)

    @staticmethod
    def _load_vocabs_file(vocab_file):
        vocab = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())
        return vocab, vocab_size

    @staticmethod
    def _load_embedding_file(embedding_file):
        embedding_dict = dict()
        embedding_size = None
        with codecs.getreader("utf-8")(tf.gfile.GFile(embedding_file, "rb")) as f:
            for line in f:
                tokens = line.strip().split(" ")
                word = tokens[0]
                vec = list(map(float, tokens[1:]))
                embedding_dict[word] = vec
                if embedding_size:
                    assert embedding_size == len(vec), "All embedding size should be same"
                else:
                    embedding_size = len(vec)
        return embedding_dict, embedding_size
