from src.tokenizer import CharacterTokenizer

def test_tokenizer_encode_decode():
    dados_treino = "olá mundo, este é o modelo do kaio!"
    tokenizer = CharacterTokenizer(dados_treino)
    
    texto_original = "olá kaio"
    ids = tokenizer.encode(texto_original)
    texto_recuperado = tokenizer.decode(ids)
    
    assert texto_original == texto_recuperado
    assert len(ids) == len(texto_original)