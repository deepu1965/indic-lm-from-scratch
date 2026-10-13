import unittest
import unicodedata
from phase3.scripts.evaluation.exact_match_normalizer import normalize_for_match

class TestExactMatchNormalizer(unittest.TestCase):
    def test_identical_strings(self):
        s1 = "सिमरन की मात्रा कम है।"
        s2 = "सिमरन की मात्रा कम है।"
        self.assertEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_leading_trailing_whitespace(self):
        s1 = "  सिमरन की मात्रा कम है।  "
        s2 = "सिमरन की मात्रा कम है।"
        self.assertEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_repeated_whitespace(self):
        s1 = "सिमरन  की   मात्रा कम  है।"
        s2 = "सिमरन की मात्रा कम है।"
        self.assertEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_unicode_normalization(self):
        # NFD vs NFC using character that decomposes
        # U+0921 (ड) + U+093C (़) = U+095C (ड़)
        s1 = "\u095c" # NFC composed 'ड़'
        s2 = "\u0921\u093c" # NFD decomposed 'ड' + '़'
        self.assertNotEqual(s1, s2)
        self.assertEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_different_entity(self):
        s1 = "करण की मात्रा कम है।"
        s2 = "सिमरन की मात्रा कम है।"
        self.assertNotEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_different_relation_word(self):
        s1 = "सिमरन की मात्रा अधिक है।"
        s2 = "सिमरन की मात्रा कम है।"
        self.assertNotEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_extra_meaningful_content(self):
        s1 = "सिमरन की मात्रा कम है। यह सही है।"
        s2 = "सिमरन की मात्रा कम है।"
        self.assertNotEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_hindi_example(self):
        # Hindi target text
        s1 = "सुमित की उम्र कम है।"
        s2 = "  सुमित की उम्र कम है। \n"
        self.assertEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_nepali_example(self):
        # Nepali target text
        s1 = "करणको मात्रा कम छ।"
        s2 = "करणको  मात्रा कम छ।\t"
        self.assertEqual(normalize_for_match(s1), normalize_for_match(s2))
        
    def test_no_accidental_match(self):
        # Ensure punctuation is NOT stripped making these equal
        s1 = "सिमरन की मात्रा कम है"
        s2 = "सिमरन की मात्रा कम है।"
        self.assertNotEqual(normalize_for_match(s1), normalize_for_match(s2))

if __name__ == "__main__":
    unittest.main()
