import abc
import tensorflow as tf


class EncoderInterface(abc.ABC):
    """Encoder interface."""

    @abc.abstractmethod
    def encode(self, sequence_inputs, sequence_length, mode):
        """Encode source inputs.

        Args:
          mode: mode
          sequence_inputs: A tensor, embedding representation of inputs sequence
          sequence_length: A tensor, input sequences' length

        Returns:
          encoder_outputs: A tensor, outputs of encoder
          encoder_state: A tensor, states of encoder
        """
        raise NotImplementedError()


class AbstractEncoder(EncoderInterface):
    def __init__(self, params, scope="encoder", dtype=tf.float32):
        """Init abstract encoder.

        Args:
          params: A python object, hparams
          scope: A constant string, variables scope
          dtype: A constant, variables dtype
        """

        self.scope = scope
        self.dtype = dtype
        self.num_encoder_layers = params["num_encoder_layers"]
        self.num_encoder_residual_layers = params["num_encoder_residual_layers"]
        self.encoder_type = params["encoder_type"]
        self.time_major = params["time_major"]
        self.unit_type = params["unit_type"]
        self.num_units = params["num_units"]
        self.forget_bias = params["forget_bias"]
        self.dropout = params["dropout"]

    def encode(self, sequence_inputs, sequence_length, mode):
        '''
        抽象编码做一些准备工作
        :param sequence_inputs:
        :param sequence_length:
        :param mode:
        :return:
        '''
        num_layers = self.num_encoder_layers
        num_residual_layers = self.num_encoder_residual_layers

        with tf.variable_scope(self.scope, dtype=tf.float32, reuse=tf.AUTO_REUSE):

            if self.time_major:
                sequence_inputs = tf.transpose(sequence_inputs, perm=[1, 0, 2])

            if self.encoder_type == "uni":
                cell = self._build_encoder_cell(mode, num_layers, num_residual_layers)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=sequence_inputs,
                    dtype=self.dtype,
                    sequence_length=sequence_length,
                    time_major=self.time_major,
                    swap_memory=True)
            elif self.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)
                num_bi_residual_layers = int(num_residual_layers / 2)
                fw_cell, bw_cell = self._build_bidirectional_encoder_cell(
                    mode=mode,
                    num_bi_layers=num_bi_layers,
                    num_bi_residual_layers=num_bi_residual_layers)
                bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell,
                    cell_bw=bw_cell,
                    inputs=sequence_inputs,
                    dtype=self.dtype,
                    sequence_length=sequence_length,
                    time_major=self.time_major,
                    swap_memory=True)
                encoder_outputs = tf.concat(bi_outputs, -1)
                # flatten states
                # ((state_fw0, states_bw0), (states_fw1, states_bw1) ->
                # (states_fw0, states_bw0, states_fw1, states_bw1)
                encoder_state = []
                for layer_id in range(num_bi_layers):
                    encoder_state.append(bi_encoder_state[0][layer_id])
                    encoder_state.append(bi_encoder_state[1][layer_id])
                encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Invalid encoder type: %s" % self.encoder_type)
            return encoder_outputs, encoder_state

    def _build_encoder_cell(self, mode, num_layers, num_residual_layers):
        """Create encoder cells.

           Args:
             mode: mode
             num_layers: A integer, number of layers
             num_residual_layers: A integer, number of residual layers

           Returns:
             Encoder's rnn cells.
           """
        raise NotImplementedError()

    def _build_bidirectional_encoder_cell(self, mode, num_bi_layers, num_bi_residual_layers):
        """Create bi-directional cells.

           Args:
             mode: mode
             num_bi_layers: A integer, number of bidirectional layers
             num_bi_residual_layers: A integer, number of bidirectional residual layers

           Returns:
             Encoder's forward and backward rnn cells.
           """
        forward_cell = self._build_encoder_cell(
            mode=mode,
            num_layers=num_bi_layers,
            num_residual_layers=num_bi_residual_layers)
        backward_cell = self._build_encoder_cell(
            mode=mode,
            num_layers=num_bi_layers,
            num_residual_layers=num_bi_residual_layers)
        return forward_cell, backward_cell


class NMTEncoder(AbstractEncoder):

    def __init__(self, params, scope="encoder", dtype=tf.float32):
        super().__init__(params, scope, dtype)

    def _build_encoder_cell(self, mode, num_layers, num_residual_layers):
        '''
        构建编码cell
        :param mode:
        :param num_layers:
        :param num_residual_layers:
        :return: Encoder's rnn cells.
        '''
        cells = []
        for i in range(num_layers):
            residual = (i >= num_layers - num_residual_layers)
            cell = self._build_single_cell(
                unit_type=self.unit_type,
                num_units=self.num_units,
                forget_bias=self.forget_bias,
                dropout=self.dropout,
                mode=mode,
                residual_conn=residual,
                residual_fn=None)
            cells.append(cell)
        # cells = cells[0] if len(cells) == 1 else cells
        return tf.nn.rnn_cell.MultiRNNCell(cells)

    @staticmethod
    def _build_single_cell(unit_type, num_units, forget_bias, dropout, mode, residual_conn, residual_fn):
        '''
         build single cell
        :param unit_type:
        :param num_units:
        :param forget_bias:
        :param dropout:
        :param mode:
        :param residual_conn:
        :param residual_fn:
        :return: A RNNCell or it's subclass
        '''
        dropout = dropout if mode != tf.estimator.ModeKeys.PREDICT else 0.0
        if unit_type == "lstm":
            single_cell = tf.nn.rnn_cell.LSTMCell(
                num_units, forget_bias=forget_bias)
        elif unit_type == "gru":
            single_cell = tf.nn.rnn_cell.GRUCell(num_units)
        elif unit_type == "layer_norm_lstm":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units, forget_bias=forget_bias, layer_norm=True)
        elif unit_type == "nas":
            single_cell = tf.contrib.rnn.NASCell(num_units)
        else:
            raise ValueError("Invalid unit type: %s" % unit_type)
        if dropout > 0.0:
            single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, 1.0 - dropout)
        if residual_conn:
            single_cell = tf.nn.rnn_cell.ResidualWrapper(single_cell, residual_fn)
        return single_cell


class GNMTEncoder(AbstractEncoder):
    def _build_encoder_cell(self, mode, num_layers, num_residual_layers):
        pass


class TransformerEncoder(AbstractEncoder):
    def _build_encoder_cell(self, mode, num_layers, num_residual_layers):
        pass
