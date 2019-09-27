#!encoding=utf-8
import tensorflow as tf
from NMTmodel.NMT.embedding import NMTEmbedding
from NMTmodel.NMT.encoder import NMTEncoder
from NMTmodel.NMT.dataset import data_util
import yaml
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)

tf.enable_eager_execution()


class NMTEncoderTest(tf.test.TestCase):
    def setUp(self):
        self.params = self._build_params()
        self.embedding = NMTEmbedding(params=self.params)
        self.encoder = NMTEncoder(params=self.params)

    @staticmethod
    def _build_params():
        config_file = os.path.join(par_dir, "config.yml")
        with tf.gfile.GFile(config_file, "r+") as f:
            params = yaml.load(stream=f.read(), Loader=yaml.FullLoader)
            return params

    def test_encoder(self):
        features, labels = data_util.get_dataset(self.params, mode=tf.estimator.ModeKeys.EVAL)
        inputs = features["input"]
        inputs_length = features["input_length"]
        print("inputs:\n", inputs)
        embedding_inputs, inputs_id = self.embedding.encoder_embedding_input(inputs=inputs)
        print("inputs_id:\n", inputs_id)
        print("embed_inputs:\n", embedding_inputs)
        print("\n")
        sequence_length = inputs_length
        encoder_outputs, encoder_states = self.encoder.encode(sequence_inputs=embedding_inputs,
                                                              sequence_length=sequence_length,
                                                              mode=tf.estimator.ModeKeys.EVAL)
        print("encoder_outputs:\n", encoder_outputs)
        print("encoder_states:\n", encoder_states)


if __name__ == '__main__':
    tf.test.main()
