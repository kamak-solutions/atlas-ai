import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.tokenizer import CharacterTokenizer

# Hiperparâmetros de Treino
batch_size = 32     # Aumentamos o lote para treinar mais rápido e estável
block_size = 8      # Contexto de caracteres
max_iters = 3000    # Quantas vezes a IA vai rodar o loop de estudo
learning_rate = 1e-2 # Velocidade de ajuste dos pesos (passo do otimizador)

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_remodelados = logits.view(B*T, C)
            targets_remodelados = targets.view(B*T)
            loss = F.cross_entropy(logits_remodelados, targets_remodelados)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx é uma matriz (B, T) de índices no contexto atual
        for _ in range(max_new_tokens):
            # Obtém as previsões
            logits, loss = self(idx)
            # Foca apenas no último passo temporal (o último caractere gerado)
            logits = logits[:, -1, :] # vira (B, C)
            # Aplica softmax para transformar notas em probabilidades reais (0 a 1)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sorteia o próximo caractere com base na distribuição de probabilidade
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Alimenta o novo caractere na sequência em andamento
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def executar_treinamento():
    modelo = BigramLanguageModel(vocab_size)
    
    # Criamos o otimizador AdamW (o motor de ajuste de pesos da IA)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=learning_rate)
    
    print("--- Texto Gerado ANTES do Treino (Chute Aleatório) ---")
    contexto_inicial = torch.zeros((1, 1), dtype=torch.long) # Começa com o token 0 (geralmente espaço)
    print(tokenizer.decode(modelo.generate(contexto_inicial, max_new_tokens=100)[0].tolist()))
    print("-" * 50)

    print("\nIniciando o loop de treinamento...")
    for iteracao in range(max_iters):
        # 1. Pega um lote de treino
        xb, yb = get_batch()
        
        # 2. Roda o modelo e calcula o erro
        logits, loss = modelo(xb, yb)
        
        # 3. Zera os gradientes antigos do passo anterior (padrão do PyTorch)
        optimizer.zero_grad(set_to_none=True)
        
        # 4. Backpropagation: calcula o quanto cada peso contribuiu para o erro
        loss.backward()
        
        # 5. Atualiza os pesos na tabela
        optimizer.step()
        
        # Imprime o progresso a cada 500 passos
        if iteracao % 500 == 0:
            print(f"Passo {iteracao:4d} | Loss (Erro): {loss.item():.4f}")

    print("-" * 50)
    print(f"Treino Finalizado! Erro Final: {loss.item():.4f}")
    
    print("\n--- Texto Gerado DEPOIS do Treino (IA Aprendendo) ---")
    print(tokenizer.decode(modelo.generate(contexto_inicial, max_new_tokens=100)[0].tolist()))

if __name__ == "__main__":
    executar_treinamento()