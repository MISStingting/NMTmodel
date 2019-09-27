import unittest

import os


def build_source_vocab(file_path, name):
    vocab = []
    with open(file=file_path, mode="r+", encoding="utf-8", buffering=8192) as fin:
        file_list = fin.readlines()
    for word in file_list:
        word = word.strip("\n")
        wl = word.split(" ")
        vocab.extend(wl)
    vocab_path = os.path.join(os.path.dirname(file_path), name)
    vocab = set(vocab)
    with open(file=vocab_path, mode="w+", encoding="utf-8", buffering=8192) as fout:
        fout.write("<blank>" + "\n")
        fout.write("<s>" + "\n")
        fout.write("<\s>" + "\n")
        for w in vocab:
            fout.write(w + "\n")
    return vocab_path


class Test(unittest.TestCase):
    def test(self):
        actions = []
        with open(r"E:\naivenmt-master\NMTmodel\data\source_vocab", "rt", encoding="utf8") as f:
            for e in f:
                actions.append(e.strip().strip("\n"))

        with open(r"E:\naivenmt-master\NMTmodel\data\source_vocab", "wt", encoding="utf8") as fout:
            for e in actions:
                fout.write(e + "\n")

    def test_collect_vocabs(self):
        input_file = build_source_vocab(file_path=r"E:\LTTProject\SEQ2SEQ\data\total.ocr", name="source_vocab")
        input_file_2 = build_source_vocab(file_path=r"E:\LTTProject\SEQ2SEQ\data\total.std", name="target_vocab")


if __name__ == '__main__':
    unittest.main()
