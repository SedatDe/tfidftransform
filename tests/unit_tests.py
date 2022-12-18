import unittest

import tfidftransform.util
from tfidftransform.transformer import Transformer


class UtilTestCases(unittest.TestCase):
    def test_to_lower(self):
        pairs = {
            'ABC': 'abc',
            'A': 'a',
            'a': 'a',
            'B': 'b',
            'b': 'b',
            'aBc': 'abc',
            'a B c ': 'a b c ',
            'I': 'i',
            'i': 'i'
        }
        for s, s_lower_expected in pairs.items():
            s_lower = tfidftransform.util.to_lower(s)
            self.assertEqual(s_lower_expected, s_lower, 'incorrect to_lower for language invariant')

    def test_to_lower_turkish(self):
        pairs = {
            'ABC': 'abc',
            'A': 'a',
            'a': 'a',
            'B': 'b',
            'b': 'b',
            'aBc': 'abc',
            'a B c ': 'a b c ',
            'I': 'ı',
            'İ': 'i',
            'ı': 'ı',
            'i': 'i'
        }
        for s, s_lower_expected in pairs.items():
            s_lower = tfidftransform.util.to_lower(s, 'turkish')
            self.assertEqual(s_lower_expected, s_lower, 'incorrect to_lower for turkish')


class TransformerTestCases(unittest.TestCase):
    def test_null_input(self):
        transformer = Transformer()
        with self.assertRaisesRegex(ValueError, 'Texts null'):
            transformer.train(None)
        with self.assertRaisesRegex(ValueError, 'Texts null'):
            transformer.infer(None)

    def test_empty_input(self):
        transformer = Transformer()
        with self.assertRaisesRegex(ValueError, 'Texts empty'):
            transformer.train([])
        with self.assertRaisesRegex(ValueError, 'Texts empty'):
            transformer.infer([])

    def test_train_inference_order(self):
        transformer = Transformer()
        with self.assertRaisesRegex(ValueError, 'Training is not done yet'):
            transformer.infer(['hello', 'how are you'])

    def test_train_vocabulary_order(self):
        transformer = Transformer()
        with self.assertRaisesRegex(ValueError, 'Training is not done yet'):
            transformer.get_vocabulary()

    def test_vocabulary(self):
        texts = [
            'hello',
            'how are you',
            'how are you doing',
            'nice to meet you',
            'meet you again'
        ]
        vocab_expected = set(s for t in texts for s in t.split())

        transformer = Transformer()
        transformer.train(texts)
        vocab = set(transformer.get_vocabulary().keys())

        self.assertSetEqual(vocab_expected, vocab, 'incorrect transformer vocabulary')

    def test_vocabulary_with_stopwords(self):
        texts = [
            'hello',
            'how are you',
            'how are you doing',
            'nice to meet you',
            'meet you again'
        ]
        vocab_expected = set(s for t in texts for s in t.split())

        stop_words = ['to', 'are']
        for sw in stop_words:
            vocab_expected.remove(sw)

        transformer = Transformer(stop_words=stop_words)
        transformer.train(texts)
        vocab = set(transformer.get_vocabulary().keys())

        self.assertSetEqual(vocab_expected, vocab, 'incorrect transformer vocabulary')

    def test_vocabulary_with_lowercase(self):
        texts = [
            'hello',
            'How are YOU',
            'how are you doing',
            'nice to meet you',
            'meet you again'
        ]
        vocab_expected = set(s.lower() for t in texts for s in t.split())

        transformer = Transformer(convert_to_lower=True)
        transformer.train(texts)
        vocab = set(transformer.get_vocabulary().keys())

        self.assertSetEqual(vocab_expected, vocab, 'incorrect transformer vocabulary')

    def test_train(self):
        texts = [
            'this is first text',
            'this is second text',
            'this would be third text'
        ]

        transformer = Transformer()
        train_result = transformer.train(texts)

        vocab = transformer.get_vocabulary()

        self.assertGreater(train_result[0, vocab['this']], 0.0)
        self.assertGreater(train_result[1, vocab['this']], 0.0)
        self.assertGreater(train_result[2, vocab['this']], 0.0)

        self.assertGreater(train_result[0, vocab['is']], 0.0)
        self.assertGreater(train_result[1, vocab['is']], 0.0)
        self.assertAlmostEqual(train_result[2, vocab['is']], 0.0)

        self.assertAlmostEqual(train_result[0, vocab['third']], 0.0)
        self.assertAlmostEqual(train_result[1, vocab['third']], 0.0)
        self.assertGreater(train_result[2, vocab['third']], 0.0)

    def test_inference(self):
        texts_train = [
            'this is first',
            'this is second',
            'this is third'
        ]

        texts_inference = ['this second']

        transformer = Transformer()
        transformer.train(texts_train)

        vocab = transformer.get_vocabulary()

        inference_result = transformer.infer(texts_inference)

        self.assertGreater(
            inference_result[0, vocab['second']],
            inference_result[0, vocab['this']]
        )


if __name__ == '__main__':
    unittest.main()
