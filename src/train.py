import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.tokenizer import CharacterTokenizer

# Hiperparâmetros de nível GPT
batch_size = 32
block_size = 16
max_iters = 3000
learning_rate = 1e-3
eval_interval = 500
eval_iters = 200
n_embd = 64      # Aumentamos os canais internos para suportar as 4 cabeças
n_head = 4       # 4 cabeças de atenção trabalhando juntas em paralelo!

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

n = int(0.9 * len(dados_tensor))
train_data = dados_tensor[:n]
val_data = dados_tensor[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(modelo):
    out = {}
    modelo.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = modelo(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    modelo.train()
    return out

# --- 1. UMA CABEÇA DE ATENÇÃO ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

# --- 2. MULTI-HEAD ATTENTION (VÁRIAS CABEÇAS EM PARALELO) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Criamos uma lista de cabeças usando ModuleList do PyTorch
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projeção linear para unificar a saída das cabeças
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Roda cada cabeça de atenção individualmente e concatena as saídas nos canais (C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# --- 3. FEED-FORWARD NETWORK (A CAMADA DE PENSAMENTO) ---
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projeção de volta
        )

    def forward(self, x):
        return self.net(x)

# --- 4. O BLOCO TRANSFORMER COMPLETO ---
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Conexões residuais somando o X original ao resultado normalizado (LayerNorm)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- ARQUITETURA FINAL DO NOSSO DECODER TRANSFORMER ---
class AtlasGPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # O nosso cérebro agora possui um bloco completo do Transformer!
        self.transformer_block = Block(n_embd, n_head=n_head)
        
        self.ln_f = nn.LayerNorm(n_embd) # Camada de normalização final
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb 
        
        # Passa pelo bloco Transformer completo (Atenção + Neurônios)
        x = self.transformer_block(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def executar_treinamento():
    modelo = AtlasGPTModel(vocab_size)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=learning_rate)
    
    print("Iniciando o Treino do AtlasGPT (Transformer Multi-Head)...")
    for iteracao in range(max_iters):
        if iteracao % eval_interval == 0:
            losses = estimate_loss(modelo)
            print(f"Passo {iteracao:4d} | Loss Treino: {losses['train']:.4f} | Loss Validação: {losses['val']:.4f}")
            
        xb, yb = get_batch('train')
        logits, loss = modelo(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(modelo)
    print("-" * 60)
    print(f"Treino Finalizado! Loss Treino Final: {losses['train']:.4f} | Loss Val Final: {losses['val']:.4f}")
    print("-" * 60)
    
    # --- NOVO: SALVANDO OS PESOS DO MODELO EM DISCO ---
    print("Salvando o cérebro da IA em 'data/atlas_gpt_pesos.pth'...")
    torch.save(modelo.state_dict(), "data/atlas_gpt_pesos.pth")
    print("Modelo salvo com sucesso!")
    
    contexto_inicial = torch.zeros((1, 1), dtype=torch.long)
    print("\n--- Texto Gerado Pelo AtlasGPT Final ---")
    print(tokenizer.decode(modelo.generate(contexto_inicial, max_new_tokens=150)[0].tolist()))
    modelo = AtlasGPTModel(vocab_size)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=learning_rate)
    
    print("Iniciando o Treino do AtlasGPT (Transformer Multi-Head)...")
    for iteracao in range(max_iters):
        if iteracao % eval_interval == 0:
            losses = estimate_loss(modelo)
            print(f"Passo {iteracao:4d} | Loss Treino: {losses['train']:.4f} | Loss Validação: {losses['val']:.4f}")
            
        xb, yb = get_batch('train')
        logits, loss = modelo(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(modelo)
    print("-" * 60)
    print(f"Treino Finalizado! Loss Treino Final: {losses['train']:.4f} | Loss Val Final: {losses['val']:.4f}")
    print("-" * 60)
    
    contexto_inicial = torch.zeros((1, 1), dtype=torch.long)
    print("\n--- Texto Gerado Pelo AtlasGPT Final ---")
    print(tokenizer.decode(modelo.generate(contexto_inicial, max_new_tokens=150)[0].tolist()))

if __name__ == "__main__":
    executar_treinamento()