import tensorflow as tf


# tf.enable_eager_execution()


class Dataset(object):

    def get_dataset(self, params, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            features_path = params["train_features_file"]
            labels_path = params["train_labels_file"]
        elif mode == tf.estimator.ModeKeys.EVAL:
            features_path = params["eval_features_file"]
            labels_path = params["eval_labels_file"]

        elif mode == tf.estimator.ModeKeys.PREDICT:
            features_path = params["test_features_file"]
            labels_path = params["test_labels_file"]
        else:
            raise ValueError("wrong mode!!!")

        features_dataset, labels_dataset = self._load_dataset(features_path, labels_path, mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = features_dataset.map(lambda x: tf.string_split([x]).values)
            dataset = dataset.shuffle(buffer_size=params["buffer_size"],
                                      reshuffle_each_iteration=params["reshuffle_each_iteration"])
            dataset = dataset.prefetch(buffer_size=params["buffer_size"])
            dataset = dataset.map(lambda src: (src, tf.size(src)))
            dataset = dataset.padded_batch(batch_size=params["batch_size"],
                                           padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
                                           padding_values=(tf.constant("<blank>"), 0))
            iterator = dataset.make_one_shot_iterator()
            src, src_len = iterator.get_next()
            features = {
                "input": src,
                "input_length": src_len
            }
            labels = None
        else:
            dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
            dataset = dataset.map(lambda x, y: (tf.string_split([x]).values, tf.string_split([y]).values))
            dataset = dataset.repeat(params["repeat"]).shuffle(buffer_size=params["buffer_size"],
                                                               reshuffle_each_iteration=params[
                                                                   "reshuffle_each_iteration"])
            dataset = dataset.prefetch(buffer_size=params["buffer_size"])
            if params["src_max_len"] > 0:
                dataset = dataset.map(
                    lambda src, tgt: (src[:params["src_max_len"]], tgt))
            if params["tgt_max_len"] > 0:
                dataset = dataset.map(
                    lambda src, tgt: (src, tgt[:params["tgt_max_len"]]))
            dataset = dataset.map(
                lambda src, tgt: (src,
                                  tf.concat((["<s>"], tgt), 0),
                                  tf.concat((tgt, ["</s>"]), 0)),
                num_parallel_calls=params["num_parallel_calls"])
            dataset = dataset.map(lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_out)))
            dataset = dataset.padded_batch(batch_size=params["batch_size"],
                                           padded_shapes=(
                                               tf.TensorShape([None]),
                                               tf.TensorShape([None]),
                                               tf.TensorShape([None]),
                                               tf.TensorShape([]),
                                               tf.TensorShape([])),
                                           padding_values=(
                                               tf.constant("<blank>", dtype=tf.string),
                                               tf.constant("<s>", dtype=tf.string),
                                               tf.constant("</s>", dtype=tf.string),
                                               0,
                                               0))
            iterator = dataset.make_one_shot_iterator()
            src, tgt_in, tgt_out, input_length, output_length = iterator.get_next()
            features = {
                "input": src,
                "input_length": input_length
            }
            labels = {
                "output_in": tgt_in,
                "output_out": tgt_out,
                "output_length": output_length
            }
        return features, labels

    @staticmethod
    def _load_dataset(features_path, labels_path, mode):
        ''' 从文件读取dataset
        :param mode:
        :return:
        '''
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            features_dataset = tf.data.TextLineDataset(filenames=features_path)
            labels_dataset = tf.data.TextLineDataset(filenames=labels_path)

            return features_dataset, labels_dataset
        elif mode == tf.estimator.ModeKeys.PREDICT:
            features_dataset = tf.data.TextLineDataset(filenames=features_path)
            return features_dataset, None


data_util = Dataset()
