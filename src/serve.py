import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.train import AtlasGPTModel, carregar_dados

def iniciar_chat():
    print("Carregando o vocabulário e os dados do Tokenizer...")
    dados_tensor, tokenizer = carregar_dados()
    vocab_size = tokenizer.vocab_size
    
    # Caminho dos pesos
    pesos_path = "data/atlas_gpt_pesos.pth"
    if not os.path.exists(pesos_path):
        print(f"Erro: O arquivo de pesos '{pesos_path}' não foi encontrado!")
        print("Execute o treinamento primeiro rodando o container de treino.")
        sys.exit(1)
        
    print("Inicializando a arquitetura do Transformer...")
    modelo = AtlasGPTModel(vocab_size)
    
    print("Injetando os pesos treinados nos neurônios...")
    modelo.load_state_dict(torch.load(pesos_path, map_location=torch.device('cpu')))
    modelo.eval() # Modo de inferência
    
    print("\n" + "="*50)
    print("   ATLAS-AI CAPTURADO EM PRODUÇÃO - PRONTO ")
    print("="*50)
    
    # Prompt inicial simulando o cliente entrando no chat
    prompt_cliente = " Olá! "
    print(f"Prompt do Cliente (Injetado): '{prompt_cliente}'")
    
    # Tokeniza a entrada do cliente
    contexto_ids = tokenizer.encode(prompt_cliente)
    x = torch.tensor([contexto_ids], dtype=torch.long) # Formato (1, T)
    
    print("\nAtlasGPT respondendo...")
    # O modelo gera a continuação da conversa baseado na atenção do prompt!
    resposta_ids = modelo.generate(x, max_new_tokens=100)[0].tolist()
    
    print("-" * 50)
    print(tokenizer.decode(resposta_ids))
    print("-" * 50)

if __name__ == "__main__":
    iniciar_chat()