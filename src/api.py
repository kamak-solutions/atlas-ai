import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.train import AtlasGPTModel, carregar_dados

# Variáveis globais de controle
modelo = None
tokenizer = None

# Gerenciador de ciclo de vida moderno da API
@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo, tokenizer
    print("==================================================")
    print("   LIFESPAN: CARREGANDO PESOS DO ATLAS-AI...   ")
    print("==================================================")
    
    dados_tensor, tokenizer = carregar_dados()
    vocab_size = tokenizer.vocab_size
    
    pesos_path = "data/atlas_gpt_pesos.pth"
    if not os.path.exists(pesos_path):
        raise RuntimeError(f"Pesos do modelo não encontrados em {pesos_path}!")
        
    modelo = AtlasGPTModel(vocab_size)
    modelo.load_state_dict(torch.load(pesos_path, map_location=torch.device('cpu')))
    modelo.eval()
    print("-> AtlasGPT carregado com sucesso na memória do container!")
    yield
    print("Desligando API e limpando recursos...")

# Inicializa o FastAPI aplicando o ciclo de vida seguro
app = FastAPI(title="Atlas-AI Production API", version="1.0", lifespan=lifespan)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.get("/")
def health_check():
    return {"status": "online", "model": "AtlasGPT-Transformer"}

@app.post("/chat")
def gerar_resposta(request: ChatRequest):
    global modelo, tokenizer
    
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="O prompt não pode estar vazio.")
        
    try:
        prompt_original = request.prompt.strip()
        # Colocamos um espaço para simular o padrão de treino
        prompt_formatado = f" {prompt_original} "
        
        contexto_ids = tokenizer.encode(prompt_formatado)
        x = torch.tensor([contexto_ids], dtype=torch.long)
        
        # Deixamos o PyTorch gerar os tokens de forma nativa e otimizada
        with torch.no_grad(): # Desativa gradientes para economizar MUITA memória
            resposta_ids = modelo.generate(x, max_new_tokens=request.max_tokens)[0].tolist()
            
        texto_gerado = tokenizer.decode(resposta_ids).strip()
        
        # --- ENGENHARIA DE CORTE CIRÚRGICO ---
        # 1. Remove o eco da pergunta
        if texto_gerado.startswith(prompt_original):
            resposta_limpa = texto_gerado[len(prompt_original):].strip()
        else:
            resposta_limpa = texto_gerado
            
        # 2. Limpa pontuações grudadas no início
        if resposta_limpa.startswith(('?', '!', '.', ':', ',', '-')):
            resposta_limpa = resposta_limpa[1:].strip()
            
        # 3. O SEGREDO: Corta na primeira quebra de linha que encontrar!
        # Se o modelo gerou a resposta e pulou linha para inventar outra coisa, nós ignoramos o resto.
        if "\n" in resposta_limpa:
            resposta_limpa = resposta_limpa.split("\n")[0].strip()
            
        return {
            "prompt_original": prompt_original,
            "resposta_completa": resposta_limpa
        }
    except Exception as e:
        print(f"[ERRO API]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))