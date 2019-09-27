#!encoding=utf-8
import tensorflow as tf
from NMTmodel.NMT.embedding import NMTEmbedding
from NMTmodel.NMT.encoder import NMTEncoder
from NMTmodel.NMT.dataset import data_util
from NMTmodel.NMT.decoder import NMTDecoder
import yaml
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)

tf.enable_eager_execution()


class NMTDecoderTest(tf.test.TestCase):
    def setUp(self):
        self.params = self._build_params()
        self.embedding = NMTEmbedding(params=self.params)
        self.encoder = NMTEncoder(params=self.params)
        self.decoder = NMTDecoder(params=self.params, embedding=self.embedding)

    @staticmethod
    def _build_params():
        config_file = os.path.join(par_dir, "config.yml")
        with tf.gfile.GFile(config_file, "r+") as f:
            params = yaml.load(stream=f.read(), Loader=yaml.FullLoader)
            return params

    def test_decoder(self):
        features, labels = data_util.get_dataset(self.params, mode=tf.estimator.ModeKeys.EVAL)
        inputs = features["input"]
        inputs_length = features["input_length"]
        print("inputs:\n", inputs)
        print("\n")
        print("labels:\n", labels)
        embedding_inputs, inputs_id = self.embedding.encoder_embedding_input(inputs=inputs)
        print("\ninputs_id:\n", inputs_id)
        print("\nembed_inputs:\n", embedding_inputs)
        print("\n")

        sequence_length = inputs_length
        encoder_outputs, encoder_state = self.encoder.encode(sequence_inputs=embedding_inputs,
                                                             sequence_length=sequence_length,
                                                             mode=tf.estimator.ModeKeys.EVAL)
        print("\nencoder_outputs:\n", encoder_outputs)
        print("\nencoder_states:\n", encoder_state)
        labels_in_embedding = self.embedding.decoder_embedding_input(inputs=labels["output_in"])
        labels_out_embedding = self.embedding.decoder_embedding_input(inputs=labels["output_out"])
        labels_len = labels["output_length"]
        # embedding target sequences str2int
        new_labels = {
            "output_in": labels_in_embedding,
            "output_out": labels_in_embedding,
            "output_length": labels_len
        }
        logits, sample_ids, dest_state = self.decoder.decode(mode=tf.estimator.ModeKeys.EVAL,
                                                             encoder_outputs=encoder_outputs,
                                                             encoder_state=encoder_state,
                                                             labels=new_labels,
                                                             src_seq_len=labels_len)
        print("logits:\n", logits)
        print("\nsample_ids:\n", sample_ids)
        print("\ndest_state[0]:\n", dest_state[0])
        print("\ndest_state[1]:\n", dest_state[1])

        # logits_and_sample_ids:
        logits_to3dim = logits[0][0]
        print("\nlogits_to3dim:\n", logits_to3dim)

        logits_matrix = tf.nn.softmax(logits)
        logits_matrix_max = tf.argmax(logits_matrix, axis=-1)
        print("\nlogits_matrix:\n", logits_matrix)
        print("\nsample_ids:\n", sample_ids)
        print("\nlogits_matrix_max:\n", logits_matrix_max)


if __name__ == '__main__':
    tf.test.main()
