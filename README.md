# 🌐 Atlas-AI — LLM Corporativo Autônomo

> **Propriedade Intelectual e Detentora:** Atlas Startup  
> **Ambiente de Desenvolvimento:** Docker Containers em Fluxo Controlado (Git Flow)  
> **Branch Atual de Engenharia:** `develop`

---

## 📖 1. Descrição do Projeto

O **Atlas-AI** é um Large Language Model (LLM) proprietário, autorregressivo e baseado em arquitetura Decoder-Only Transformer com mecanismos de Multi-Head Attention (Atenção Multi-Cabeça). Desenvolvido do zero a nível de caracteres pela **Atlas Startup**, o modelo foi concebido para resolver o gargalo de atendimento automatizado e inteligência de dados operacionais (como suporte a pagamentos via Pix e cartões corporativos).

Diferente de APIs de terceiros que oneram a operação com custos variáveis e latência externa, o Atlas-AI roda localmente em containers isolados via Docker. Isso garante soberania total sobre os dados da startup, privacidade regulatória estrutural e a flexibilidade de customização de hiperparâmetros diretamente nas matrizes de pesos (tensores do PyTorch).

### 🛠️ Stack Tecnológica & Arquitetura
* **Backend Core:** Python 3.10 & PyTorch 2.x (Manipulação de tensores e redes de atenção)
* **API de Inferência:** FastAPI & Uvicorn com hot-reload ativo
* **Orquestração e Isolamento:** Docker Engine & Docker Compose
* **Arquitetura de IA:** Multi-Head Attention com blocos de contexto expandidos e gerenciamento de memória via `with torch.no_grad()`

---

## 🚀 2. Onboarding de Engenharia (Guia de Setup Rápido)

Seja bem-vindo ao time de engenharia da **Atlas Startup**! Este guia foi desenhado para colocar o ambiente de inteligência artificial de pé na sua máquina de desenvolvimento local em menos de 5 minutos, totalmente imune a conflitos de dependências de escopo.

### 📋 Pré-requisitos do Sistema
Antes de rodar os comandos, certifique-se de possuir:
* Sistema Operacional Linux (Arch Linux/Ubuntu recomendados), macOS ou Windows via WSL2.
* Docker Engine & Docker Compose instalados.
* Git para gerenciamento de branches.

> ⚠️ **DIRETRIZ CRÍTICA DE GOVERNANÇA (GIT FLOW):** > Nunca realize commits diretamente na branch `main` ou `master`. Todo o desenvolvimento do core do modelo, testes de hiperparâmetros e endpoints de API acontecem rigorosamente sob a branch `develop` ou branches auxiliares de `feat/` e `fix/`.

---

### 💻 Passo a Passo para o Primeiro Boot

#### **Passo 1: Clonar o Repositório e Mudar para a Branch de Trabalho**
Abra o seu terminal e clone a infraestrutura da startup, apontando imediatamente para o fluxo de desenvolvimento unificado:
```bash
git clone [https://github.com/kamak-solutions/atlas-ai.git](https://github.com/kamak-solutions/atlas-ai.git)
cd atlas-ai
git checkout develop