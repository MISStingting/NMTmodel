import os
import yaml
import tensorflow as tf
from NMTmodel.NMT.dataset import data_util

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)


class DatasetTest(tf.test.TestCase):
    def setUp(self):
        self.config_file = os.path.join(par_dir, "config.yml")

    def test_dataset(self):
        with tf.gfile.GFile(self.config_file, "rb") as f:
            params = yaml.load(stream=f.read(), Loader=yaml.FullLoader)
        data_util.get_dataset(params, mode=tf.estimator.ModeKeys.PREDICT)


if __name__ == '__main__':
    tf.test.main()
