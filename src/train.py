import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.tokenizer import CharacterTokenizer

# Hiperparâmetros
batch_size = 4
block_size = 8

def carregar_dados():
    with open("data/dataset_treino.json", "r", encoding="utf-8") as f:
        dados = json.load(f)
    texto_bruto = ""
    for item in dados:
        texto_bruto += f" {item['entrada']} {item['saida']}"
    tokenizer = CharacterTokenizer(texto_bruto)
    todos_os_ids = tokenizer.encode(texto_bruto)
    return torch.tensor(todos_os_ids, dtype=torch.long), tokenizer

dados_tensor, tokenizer = carregar_dados()
vocab_size = tokenizer.vocab_size

def get_batch():
    ix = torch.randint(len(dados_tensor) - block_size, (batch_size,))
    x = torch.stack([dados_tensor[i:i+block_size] for i in ix])
    y = torch.stack([dados_tensor[i+1:i+block_size+1] for i in ix])
    return x, y

# --- ARQUITETURA DA REDE NEURAL ---
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Cada token vai indexar diretamente a tabela de probabilidades do próximo token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx e targets são matrizes (B, T) de inteiros
        logits = self.token_embedding_table(idx) # Formato (Batch, Time, Channels) (B, T, C)
        
        if targets is None:
            loss = None
        else:
            # O PyTorch espera que a dimensão dos canais (C) seja a segunda para calcular a perda (Loss)
            B, T, C = logits.shape
            logits_remodelados = logits.view(B*T, C)
            targets_remodelados = targets.view(B*T)
            
            # Calcula a Cross Entropy Loss (o quanto a IA errou no palpite)
            loss = F.cross_entropy(logits_remodelados, targets_remodelados)

        return logits, loss

def iniciar_treino():
    xb, yb = get_batch()
    
    # Inicializa o modelo
    modelo = BigramLanguageModel(vocab_size)
    logits, loss = modelo(xb, yb)
    
    print("--- Inicializando os Neurônios do Modelo ---")
    print(f"Formato dos Logits de saída (B, T, C): {logits.shape}")
    print(f"Loss inicial (Erro do Modelo): {loss.item():.4f}")
    print("\nExplicação técnica:")
    print(f"Como temos {vocab_size} caracteres possíveis, o chute puramente aleatório")
    print(f"deveria dar um Loss por volta de -ln(1/{vocab_size}) = {torch.log(torch.tensor(vocab_size)).item():.2f}")

if __name__ == "__main__":
    iniciar_treino()