import json
import torch
from src.tokenizer import CharacterTokenizer

def carregar_e_tokenizar_dados():
    # 1. Carrega o arquivo JSON
    with open("data/dataset_treino.json", "r", encoding="utf-8") as f:
        dados = json.load(f)
    
    texto_bruto = ""
    for item in dados:
        texto_bruto += f" {item['entrada']} {item['saida']}"
    
    # 2. Inicializa o Tokenizador
    tokenizer = CharacterTokenizer(texto_bruto)
    
    # 3. Transforma TODO o texto de treino em uma lista gigante de IDs
    todos_os_ids = tokenizer.encode(texto_bruto)
    
    # 4. CONVERSÃO PARA TENSOR DO PYTORCH
    # Transformamos a lista do Python em um vetor matemático de alta performance (LongTensor)
    dados_tensor = torch.tensor(todos_os_ids, dtype=torch.long)
    
    print("--- Estatísticas dos Tensores (PyTorch) ---")
    print(f"Formato do Tensor de Dados (Shape): {dados_tensor.shape}")
    print(f"Tipo do Tensor: {dados_tensor.dtype}")
    
    # 5. Criando a lógica de Contexto (X) e Alvo (Y) para a IA aprender
    # Se a IA ler os primeiros 4 números, ela precisa tentar adivinhar o 5º número.
    tamanho_bloco = 4 
    x = dados_tensor[:tamanho_bloco]
    y = dados_tensor[1:tamanho_bloco+1]
    
    print("\n--- Conceito de Entrada (X) e Saída Esperada (Y) ---")
    print(f"Se a entrada (X) for: {x.tolist()} -> representação: '{tokenizer.decode(x.tolist())}'")
    print(f"O alvo (Y) deve ser: {y.tolist()} -> representação: '{tokenizer.decode(y.tolist())}'")

if __name__ == "__main__":
    carregar_e_tokenizar_dados()