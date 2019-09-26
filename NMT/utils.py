import tensorflow as tf
import collections


class RelateUtils(object):
    @staticmethod
    def add_dict_to_collection(name, tensors_dict):
        for k, v in tensors_dict.items():
            tf.add_to_collection(name + "_keys", k)
            tf.add_to_collection(name + "_values", v)

    @staticmethod
    def get_dict_from_collection(name):
        keys = tf.get_collection(name + "_keys")
        values = tf.get_collection(name + "_values")
        return dict(zip(keys, values))

    @staticmethod
    def add_to_collection(name, tensor):
        tf.add_to_collection(name, tensor)

    @staticmethod
    def get_from_collection(name):
        return tf.get_collection(name)

    @staticmethod
    def _format_text(words):
        """
        Convert a sequence words into sentence.
        """
        if (not hasattr(words, "__len__") and  # for numpy array
                not isinstance(words, collections.Iterable)):
            words = [words]
        return b" ".join(words)

    @staticmethod
    def _format_bpe_text(symbols, delimiter=b"@@"):
        words = []
        word = b""
        if isinstance(symbols, str):
            symbols = symbols.encode()
        delimiter_len = len(delimiter)
        for symbol in symbols:
            if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
                word += symbol[:-delimiter_len]
            else:  # end of a word
                word += symbol
                words.append(word)
                word = b""
        return b" ".join(words)

    def _format_spm_text(self, symbols):
        """
        Decode a text in SPM (https://github.com/google/sentencepiece) format
        """
        return u"".join(self._format_text(symbols).decode("utf-8").split()).replace(
            u"\u2581", u" ").strip().encode("utf-8")

    # 获取一批预测结果
    def get_predictions(self, predictions, tgt_eos, subword_option):
        """Decode the models' output tensor to text.

        Args:
          predictions: predictions, instance
          tgt_eos: target sentence's eod-of-sentence symbol.
          subword_option: subword option

        Returns:
          Text of prediction result.
        """
        if tgt_eos:
            tgt_eos = tgt_eos.encode("utf8")

        # Select first sentence
        output = predictions[0]["predict_labels"]
        print(output[0, :])
        output = output[0, :].tolist()

        if tgt_eos and tgt_eos in output:
            output = output[:output.index(tgt_eos)]

        if subword_option == "bpe":  # BPE
            translation = self._format_bpe_text(output)
        elif subword_option == "spm":  # SPM
            translation = self._format_spm_text(output)
        else:
            translation = self._format_text(output)
        return translation


utils = RelateUtils()
