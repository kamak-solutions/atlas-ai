import json
import torch
from src.tokenizer import CharacterTokenizer

# Hiperparâmetros da nossa IA
batch_size = 4  # Quantas sequências processadas em paralelo?
block_size = 8  # Qual o tamanho do contexto de letras que a IA olha?

def carregar_dados():
    with open("data/dataset_treino.json", "r", encoding="utf-8") as f:
        dados = json.load(f)
    
    texto_bruto = ""
    for item in dados:
        texto_bruto += f" {item['entrada']} {item['saida']}"
        
    tokenizer = CharacterTokenizer(texto_bruto)
    todos_os_ids = tokenizer.encode(texto_bruto)
    return torch.tensor(todos_os_ids, dtype=torch.long), tokenizer

# Carrega os tensores globais
dados_tensor, tokenizer = carregar_dados()

# FUNÇÃO DEV/AI: Gera um lote de dados com Entradas (X) e Alvos (Y)
def get_batch():
    # Sorteia índices aleatórios no texto, garantindo que caiba o block_size
    ix = torch.randint(len(dados_tensor) - block_size, (batch_size,))
    
    # Monta as linhas de contexto (X) e os alvos deslocados (Y)
    x = torch.stack([dados_tensor[i:i+block_size] for i in ix])
    y = torch.stack([dados_tensor[i+1:i+block_size+1] for i in ix])
    return x, y

def iniciar_arquitetura():
    xb, yb = get_batch()
    
    print("--- Dimensões dos Lotes de Treino ---")
    print(f"Formato do Lote de Entrada (Xb) [Batch, Block]: {xb.shape}")
    print(f"Formato do Lote de Saída (Yb) [Batch, Block]: {yb.shape}")
    
    print("\n--- Analisando o Lote por Dentro ---")
    print("Matriz Xb (Números que a IA vê):")
    print(xb)
    
    print("\nExemplo da primeira linha do lote decodificada:")
    print(f"Contexto: '{tokenizer.decode(xb[0].tolist())}'")
    print(f"Alvo Esperado: '{tokenizer.decode(yb[0].tolist())}'")

if __name__ == "__main__":
    iniciar_arquitetura()