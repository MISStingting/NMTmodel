import unittest


class Test(unittest.TestCase):
    def test(self):
        actions = []
        with open(r"E:\naivenmt-master\NMTmodel\data\source_vocab", "rt", encoding="utf8") as f:
            for e in f:
                actions.append(e.strip().strip("\n"))

        with open(r"E:\naivenmt-master\NMTmodel\data\source_vocab", "wt", encoding="utf8") as fout:
            for e in actions:
                fout.write(e + "\n")


if __name__ == '__main__':
    unittest.main()
