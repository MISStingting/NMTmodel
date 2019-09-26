#!encoding=utf-8
import tensorflow as tf
from NMTmodel.NMT.embedding import NMTEmbedding
from NMTmodel.NMT.dataset import data_util
import yaml
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)

tf.enable_eager_execution()


class EmbeddingTest(tf.test.TestCase):
    def setUp(self):
        self.params = self._build_params()
        self.embedding = NMTEmbedding(params=self.params)

    @staticmethod
    def _build_params():
        config_file = os.path.join(par_dir, "config.yml")
        with tf.gfile.GFile(config_file, "r+") as f:
            params = yaml.load(stream=f.read(), Loader=yaml.FullLoader)
            return params

    def test_str2int_embedding(self):
        features, labels = data_util.get_dataset(self.params, mode=tf.estimator.ModeKeys.EVAL)
        inputs = features["input"]
        print("inputs:\n", inputs)
        embedding_inputs = self.embedding.encoder_embedding_input(inputs=inputs)
        print("embedding_inputs:\n", embedding_inputs)
        print("\n")
        outputs_in = labels["output_in"]
        output_out = labels["output_out"]
        embedding_outputs_in = self.embedding.decoder_embedding_input(inputs=outputs_in)
        embedding_outputs_out = self.embedding.decoder_embedding_input(inputs=output_out)
        print(embedding_outputs_in)
        print(embedding_outputs_out)

    def test_int2str_embedding(self):
        '''
        预测时使用
        :return:
        '''
        features, labels = data_util.get_dataset(self.params, mode=tf.estimator.ModeKeys.PREDICT)
        inputs = features["input"]
        print("inputs:\n", inputs)
        embedding_inputs, inputs_id = self.embedding.encoder_embedding_input(inputs=inputs)
        print("embedding_inputs:\n", embedding_inputs)
        print("\n")
        labels = self.embedding.tgt_idx2str_table.lookup(keys=inputs_id)
        print("labels:", labels)


if __name__ == '__main__':
    tf.test.main()
