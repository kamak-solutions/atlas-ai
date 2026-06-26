import json
import os

def gerar_dataset_atendimento():
    # Base de dados genérica que simula múltiplos negócios
    templates = [
        {
            "contexto": "Suporte Geral",
            "perguntas": ["Olá", "Oi", "Bom dia", "Boa tarde", "Gostaria de ajuda"],
            "respostas": ["Olá! Como posso ajudar você hoje?", "Oi! Em que posso ser útil?"]
        },
        {
            "contexto": "Formas de Pagamento",
            "perguntas": ["Quais as formas de pagamento?", "Como posso pagar?", "Aceita cartão?", "Aceita Pix?"],
            "respostas": ["Aceitamos cartões de crédito, débito e Pix.", "Você pode pagar via Pix ou cartões de crédito/débito."]
        },
        {
            "contexto": "Encerramento",
            "perguntas": ["Obrigado", "Valeu", "Tchau", "Até logo"],
            "respostas": ["Por nada! Se precisar de algo mais, estou à disposição.", "Até logo! Tenha um excelente dia."]
        }
    ]
    
    dados_finais = []
    
    # Cruza perguntas e respostas para criar variações de treino
    for t in templates:
        for p in t["perguntas"]:
            for r in t["respostas"]:
                dados_finais.append({
                    "contexto": t["contexto"],
                    "entrada": p,
                    "saida": r
                })
                
    # Cria uma pasta para os dados se não existir
    os.makedirs("data", exist_ok=True)
    
    # Salva em formato JSON estruturado
    with open("data/dataset_treino.json", "w", encoding="utf-8") as f:
        json.dump(dados_finais, f, ensure_ascii=False, indent=4)
        
    print(f"Dataset gerado com sucesso! {len(dados_finais)} exemplos criados em 'data/dataset_treino.json'.")

if __name__ == "__main__":
    gerar_dataset_atendimento()