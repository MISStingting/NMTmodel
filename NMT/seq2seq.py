import abc
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from .dataset import data_util

tf.enable_eager_execution()


class AbstractModel(abc.ABC):
    def input_fn(self, params, mode):
        raise NotImplementedError()

    def model_fn(self, features, labels, mode, params, config=None):
        raise NotImplementedError()

    @staticmethod
    def serving_input_receiver_fn():
        receiver_tensors = {
            "inputs": tf.placeholder(dtype=tf.string, shape=(None, None)),
            "inputs_length": tf.placeholder(dtype=tf.int32, shape=(None,))
        }
        features = receiver_tensors.copy()
        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=receiver_tensors)


class Seq2SeqModel(AbstractModel):
    def __init__(self,
                 embedding,
                 encoder,
                 decoder,
                 scope="seq2seq",
                 dtype=tf.float32):
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.scope = scope
        self.dtype = dtype

    def input_fn(self, params, mode):
        return data_util.get_dataset(params, mode)

    def model_fn(self, features, labels, mode, params, config=None):
        src = features["input"]
        src_len = features["input_length"]
        # embedding source input
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            sequence_inputs, input_ids = self.embedding.encoder_embedding_input(src)
            encoder_outputs, encoder_states = self.encoder.encode(sequence_inputs=sequence_inputs,
                                                                  sequence_length=src_len,
                                                                  mode=mode)
            if params["time_major"]:
                encoder_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
            # embedding target sequence
            tgt_in, tgt_in_ids = self.embedding.encoder_embedding_input(labels["output_in"])
            tgt_out, tgt_out_ids = self.embedding.encoder_embedding_input(labels["output_out"])
            tgt_len = labels["output_length"]
            new_labels = {
                "output_in": tgt_in,
                "output_out": tgt_out,
                "output_length": tgt_len
            }
            # decoder
            logits, predict_ids, des_states = self.decoder.decode(mode=mode,
                                                                  encoder_outputs=encoder_outputs,
                                                                  encoder_state=encoder_states,
                                                                  labels=new_labels,
                                                                  src_seq_len=src_len)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self._build_predictions(params, predict_ids)
            tf.add_to_collections("predictions", predictions)
            key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            export_outputs = {
                key: predictions
            }
            prediction_hooks = self._build_prediction_hooks()
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs,
                                              prediction_hooks=prediction_hooks)
        loss = self.compute_loss(logits, new_labels, params)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = self._build_train_op(mode, params, loss)
            training_hooks = []
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                              training_hooks=training_hooks)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self._build_eval_metric(predict_ids, labels, src_len)
            evaluation_hooks = []
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops,
                                              evaluation_hooks=evaluation_hooks)

    @staticmethod
    def _build_predictions(params, predict_ids):
        table = lookup_ops.index_to_string_table_from_file(vocabulary_file=params["target_vocab"],
                                                           default_value="<blank>")
        predict_labels = table.lookup(keys=tf.cast(predict_ids, tf.int64))
        predictions = {
            "predict_ids": predict_ids,
            "predict_labels": predict_labels
        }
        return predictions

    @staticmethod
    def _build_prediction_hooks():
        return []

    def compute_loss(self, logits, new_labels, params):
        """
        计算损失
        :param logits: 模型的输出
        :param new_labels: 真实的标签
        :param params:
        :return:
        """
        actual_labels = new_labels["output_out"]
        print("\nactual_labels:\n", actual_labels)
        batch_size, max_time_steps = tf.shape(actual_labels)[0], tf.shape(actual_labels)[1]
        if params["time_major"]:
            actual_labels = tf.transpose(actual_labels, perm=[1, 0, 2])
            max_time_steps = tf.shape(actual_labels)[0]
            batch_size = tf.shape(actual_labels)[1]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=actual_labels, logits=logits)
        target_weights = tf.sequence_mask(
            lengths=new_labels['output_length'],
            maxlen=max_time_steps,
            dtype=self.dtype)
        loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(
            batch_size)
        return loss

    def _build_train_op(self, mode, params, loss):
        optimizer = params["optimizer"]
        if optimizer == "sgd":
            self.sgd_lr = tf.constant(params["learning_rate"])
            opt = tf.train.GradientDescentOptimizer(self.sgd_lr)
        elif params.optimizer == "adam":
            opt = tf.train.AdamOptimizer()
        else:
            raise ValueError("Unknown optimizer %s" % params.optimizer)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(loss, trainable_params)
        clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=params["max_gradient_norm"])
        grads_and_vars = zip(clipped_grads, trainable_params)
        train_op = opt.apply_gradients(grads_and_vars, tf.train.get_or_create_global_step())
        return train_op

    @staticmethod
    def _build_eval_metric(predict_ids, labels, src_len):
        '''
        构建评估矩阵，？？评估用准确度来衡量 actual_ids 元素是int
        :param predict_ids: 模型的输出
        :param labels:    实际的labels
        :param src_len: 实际labels的长度
        :param params:
        :return:
        '''
        actual_ids = labels['output_in']
        weights = tf.sequence_mask(src_len)
        metrics = {
            "accuracy": tf.metrics.accuracy(labels=actual_ids, predictions=predict_ids, weights=weights)
        }
        return metrics
