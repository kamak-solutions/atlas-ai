import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.tokenizer import CharacterTokenizer

# Hiperparâmetros
batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-2
eval_interval = 500  # A cada 500 passos, vamos avaliar o modelo
eval_iters = 200     # Quantos lotes usar para tirar a média do Loss

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

# --- DIVISÃO DOS DADOS (TRAIN / VAL SPLIT) ---
n = int(0.9 * len(dados_tensor)) # 90% para treino
train_data = dados_tensor[:n]
val_data = dados_tensor[n:]

# Atualizamos o get_batch para escolher entre os dados de treino ou validação
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Função DevOps/AI para estimar o Loss sem afetar os pesos do modelo
@torch.no_grad()
def estimate_loss(modelo):
    out = {}
    modelo.eval() # Coloca o modelo em modo de avaliação
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = modelo(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # Tira a média dos blocos
    modelo.train() # Volta o modelo para o modo de treino
    return out

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
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def executar_treinamento():
    modelo = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=learning_rate)
    
    print("Iniciando o loop de treinamento com validação cruzada...")
    
    for iteracao in range(max_iters):
        
        # A cada 'eval_interval' passos, calculamos o Loss do treino e da validação
        if iteracao % eval_interval == 0:
            losses = estimate_loss(modelo)
            print(f"Passo {iteracao:4d} | Loss Treino: {losses['train']:.4f} | Loss Validação: {losses['val']:.4f}")
            
        # Treino normal (sempre usando o split 'train')
        xb, yb = get_batch('train')
        logits, loss = modelo(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Avaliação final
    losses = estimate_loss(modelo)
    print("-" * 60)
    print(f"Treino Finalizado! Loss Treino Final: {losses['train']:.4f} | Loss Val Final: {losses['val']:.4f}")
    print("-" * 60)
    
    contexto_inicial = torch.zeros((1, 1), dtype=torch.long)
    print("\n--- Amostra de texto gerado pelo modelo avaliado ---")
    print(tokenizer.decode(modelo.generate(contexto_inicial, max_new_tokens=100)[0].tolist()))

if __name__ == "__main__":
    executar_treinamento()