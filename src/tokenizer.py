class CharacterTokenizer:
    def __init__(self, text: str):
        # 1. Cria o vocabulário com todos os caracteres únicos ordenados
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 2. Cria os dicionários de mapeamento: Caractere <-> Número
        self.char_to_id = { ch: i for i, ch in enumerate(self.chars) }
        self.id_to_char = { i: ch for i, ch in enumerate(self.chars) }

    def encode(self, text: str) -> list[int]:
        # Transforma uma string de texto em uma lista de números (IDs)
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        # Transforma uma lista de números (IDs) de volta para uma string legível
        return ''.join([self.id_to_char[i] for i in ids])