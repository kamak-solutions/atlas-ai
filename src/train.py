import json
from src.tokenizer import CharacterTokenizer

def carregar_e_tokenizar_dados():
    # 1. Carrega o arquivo JSON gerado
    with open("data/dataset_treino.json", "r", encoding="utf-8") as f:
        dados = json.load(f)
    
    # 2. Junta todo o texto para criar o texto bruto de treino
    texto_bruto = ""
    for item in dados:
        texto_bruto += f" {item['entrada']} {item['saida']}"
    
    # 3. Inicializa o nosso Tokenizador com base nesse texto
    tokenizer = CharacterTokenizer(texto_bruto)
    
    print("--- Estatísticas do Nosso Modelo Inicial ---")
    print(f"Tamanho total do texto de treino: {len(texto_bruto)} caracteres.")
    print(f"Tamanho do vocabulário único (Vocab Size): {tokenizer.vocab_size} tokens.")
    
    # 4. Transforma um exemplo de teste para ver os números na tela
    exemplo = "Olá"
    ids_codificados = tokenizer.encode(exemplo)
    print(f"\nExemplo de Codificação:")
    print(f"Texto original: '{exemplo}'")
    print(f"IDs numéricos: {ids_codificados}")
    print(f"Texto decodificado de volta: '{tokenizer.decode(ids_codificados)}'")

if __name__ == "__main__":
    carregar_e_tokenizar_dados()