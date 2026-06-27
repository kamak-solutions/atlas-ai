import requests
import json

print("==================================================")
# Mensagem simples indicando o início do chat
print("         ATLAS-AI CHAT INTERATIVO          ")
print("   Digite 'sair' para encerrar a conversa. ")
print("==================================================\n")

url = "http://ai-api:8000/chat"
while True:
    pergunta = input("Você: ")
    
    if pergunta.lower().strip() == 'sair':
        print("Saindo do chat... Até mais!")
        break
        
    if not pergunta.strip():
        continue
        
    # Payload enviado para o container da API
    payload = {
        "prompt": pergunta,
        "max_tokens": 60
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        resposta = requests.post(url, data=json.dumps(payload), headers=headers)
        if resposta.status_code == 200:
            dados = resposta.json()
            print(f"AtlasGPT: {dados['resposta_completa']}\n")
        else:
            print(f"Erro na API: {resposta.status_code}\n")
    except Exception as e:
        print(f"Erro ao conectar na API: {e}\n")