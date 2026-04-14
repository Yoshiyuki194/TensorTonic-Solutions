import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {}
        self.id_to_word = {}
        
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        next_id = 4

        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)

        unique_words = sorted(set(all_words))

        for word in unique_words:
            if word in self.word_to_id:
                continue
            self.word_to_id[word] = next_id
            self.id_to_word[next_id] = word
            next_id += 1

        self.vocab_size = next_id
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        if not self.word_to_id:
            return []

        unk_id = self.word_to_id[self.unk_token]
        words = text.lower().split()
        
        return [self.word_to_id.get(word, unk_id) for word in words]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        if not self.id_to_word:
            return ""

        words = [self.id_to_word.get(i, self.unk_token) for i in ids]
        return " ".join(words)
