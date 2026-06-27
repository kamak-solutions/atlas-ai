import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.tokenizer import CharacterTokenizer

# Hiperparâmetros ajustados para o Mini-GPT
batch_size = 32
block_size = 16  # Aumentamos o contexto de 8 para 16 caracteres!
max_iters = 3000
learning_rate = 1e-3 # Reduzimos o learning rate para estabilizar a atenção
eval_interval = 500
eval_iters = 200
n_embd = 32      # Dimensão das características internas (Embedding de tamanho 32)

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

# --- MECANISMO DE ATENÇÃO (SELF-ATTENTION HEAD) ---
class Head(nn.Module):
    """ Uma cabeça de auto-atenção (Self-Attention Head) """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Cria uma máscara triangular para garantir que o modelo não espie o futuro
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Calcula as notas de afinidade (pesos de atenção)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Aplica a máscara: impede que o caractere atual olhe para as letras da frente
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # Executa a agregação ponderada dos valores
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

# --- MODELO GPT EVOLUÍDO ---
class MiniGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Agora o token passa por um Embedding de características (n_embd)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # O modelo também aprende a posição de onde a letra está na frase!
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Adicionamos a nossa cabeça de atenção à rede
        self.sa_head = Head(n_embd)
        # Camada linear final para converter os embeddings de volta em notas para o vocabulário
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Busca os embeddings de conteúdo e de posição espacial
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Passa pelo mecanismo de Auto-Atenção
        x = self.sa_head(x) # (B, T, n_embd)
        
        # Projeta os resultados para o tamanho do vocabulário
        logits = self.lm_head(x) # (B, T, vocab_size)
        
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
            # Como aumentamos o contexto, precisamos cortar o idx para caber no block_size máximo
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def executar_treinamento():
    modelo = MiniGPTLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=learning_rate)
    
    print("Iniciando o Treino do Mini-GPT com Self-Attention...")
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
    print("\n--- Texto Gerado Pela Nova Arquitetura GPT ---")
    print(tokenizer.decode(modelo.generate(contexto_inicial, max_new_tokens=150)[0].tolist()))

if __name__ == "__main__":
    executar_treinamento()