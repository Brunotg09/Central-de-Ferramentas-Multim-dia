# 🛠️ Sistema de Ferramentas Multimídia Streamlit

Uma aplicação web completa para processamento multimídia construída com Streamlit, oferecendo ferramentas para download de vídeos, transcrição de áudio, teste de velocidade de internet, remoção de fundo de imagens e conversão texto-para-fala.

## 🚀 Funcionalidades

### 🎬 **Download de Vídeos do YouTube**

- Download de vídeos em alta qualidade (MP4)
- Extração de áudio (MP3)
- Suporte a múltiplas URLs
- Lista organizada de downloads

### 🎵 **Transcrição de Áudio/Vídeo**

- Transcrição automática usando reconhecimento de fala
- Suporte a múltiplos idiomas (Português, Inglês, Espanhol, etc.)
- Geração de PDF e TXT com timestamps
- Processamento inteligente em chunks

### 🚀 **Teste de Velocidade da Internet**

- Medição de velocidade de download/upload
- Teste de ping/latência
- Histórico de testes
- Visualizações interativas

### 🖼️ **Removedor de Fundo de Imagens**

- Remoção automática de fundo usando IA
- Refinamento manual interativo
- Suporte a diversos formatos de imagem
- Download dos resultados processados

### 🔊 **Texto para Fala (TTS)**

- Conversão de texto para áudio
- Múltiplos provedores (Edge TTS, Google TTS, PyTTSX3)
- Diversas vozes disponíveis
- Reprodução e download de áudio

## 📋 Requisitos do Sistema

### 🔧 **Dependências Python**

Todas as dependências estão listadas no arquivo `requirements.txt`. Instale com:

```bash
pip install -r requirements.txt
```

### 🖥️ **Requisitos de Sistema**

#### **Windows**

- **Python 3.8+** (recomendado 3.10+)
- **FFmpeg** (obrigatório para processamento de áudio)
  - Download: https://ffmpeg.org/download.html
  - Adicione ao PATH do sistema
- **Espaço em disco**: Pelo menos 2GB livres
- **Conexão com internet**: Para downloads e alguns serviços

#### **Linux/macOS**

- **Python 3.8+**
- **FFmpeg**: Instale via package manager

  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg

  # macOS
  brew install ffmpeg
  ```

### 📦 **Dependências Opcionais**

Algumas funcionalidades são opcionais e requerem dependências extras:

- **speedtest-cli**: Para teste de velocidade
- **rembg**: Para remoção de fundo (requer modelo de IA)
- **streamlit-drawable-canvas**: Para edição manual de imagens
- **edge-tts**: Para síntese de voz mais natural
- **gtts**: Para Google Text-to-Speech
- **pyttsx3**: Para TTS offline

## 🛠️ Instalação e Configuração

### 1. **Clonar/Preparar o Projeto**

```bash
# Navegue até a pasta do projeto
cd streamlit_bundle
```

### 2. **Criar Ambiente Virtual (Recomendado)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. **Instalar Dependências**

```bash
pip install -r requirements.txt
```

### 4. **Verificar FFmpeg**

```bash
ffmpeg -version
```

Se não funcionar, adicione FFmpeg ao PATH do sistema.

### 5. **Executar a Aplicação**

```bash
streamlit run youtube.py
```

A aplicação abrirá no navegador em `http://localhost:8501`

## 🔧 Solução de Problemas

### **Erro: "ffmpeg not found"**

- Instale FFmpeg e adicione ao PATH
- No Windows: Reinicie o terminal após adicionar ao PATH

### **Erro: "Module not found"**

- Certifique-se de que instalou todas as dependências: `pip install -r requirements.txt`
- Verifique se está no ambiente virtual correto

### **Erro na remoção de fundo**

- Instale rembg: `pip install rembg`
- Pode requerer download de modelo de IA na primeira execução

### **Problemas com TTS**

- Edge TTS requer conexão com internet
- PyTTSX3 funciona offline mas voz é mais robótica

### **Vídeos do YouTube não baixam**

- yt-dlp pode precisar de atualização: `pip install -U yt-dlp`
- Alguns vídeos podem ter restrições regionais

## 📁 Estrutura do Projeto

```
streamlit_bundle/
├── youtube.py              # Aplicação principal
├── requirements.txt        # Dependências Python
├── README.md              # Este arquivo
└── canvas_bg_cache/       # Cache para imagens (criado automaticamente)
```

## 🎯 Como Usar

1. **Página Inicial**: Visão geral das ferramentas disponíveis
2. **Baixar Vídeos**: Cole URLs do YouTube, escolha vídeo ou áudio
3. **Transcrever Áudio**: Faça upload de arquivo ou use link do YouTube
4. **Teste de Velocidade**: Execute teste de internet
5. **Remover Fundo**: Faça upload de imagem para processamento
6. **Texto para Fala**: Digite texto e gere áudio

## 🔒 Segurança e Privacidade

- Arquivos temporários são automaticamente removidos
- Dados de download ficam apenas na sessão atual
- Não há armazenamento permanente de arquivos do usuário
- Conexões HTTPS para downloads externos

## 📞 Suporte

Para problemas ou dúvidas:

1. Verifique os logs de erro no terminal
2. Certifique-se de que todas as dependências estão instaladas
3. Teste com arquivos pequenos primeiro
4. Reinicie a aplicação se necessário

## 📝 Licença

Este projeto é para uso pessoal. Respeite os termos de serviço das plataformas utilizadas (YouTube, etc.).

---

**Última atualização**: Outubro 2025
