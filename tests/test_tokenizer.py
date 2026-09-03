import unittest

from src.tokenizer import CharacterTokenizer


class TestCharacterTokenizer(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        text = "to be or not to be\n"
        tokenizer = CharacterTokenizer(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        self.assertEqual(decoded, text)
        self.assertEqual(tokenizer.vocab_size, len(set(text)))

    def test_unknown_character_raises_error(self):
        tokenizer = CharacterTokenizer("abc")

        with self.assertRaises(KeyError):
            tokenizer.encode("z")


if __name__ == "__main__":
    unittest.main()
