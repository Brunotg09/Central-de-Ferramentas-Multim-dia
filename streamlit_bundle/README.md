# 🛠️ Sistema de Ferramentas Multimídia Streamlit

Uma aplicação web completa para processamento multimídia construída com Streamlit, oferecendo ferramentas para dow### **2. Limitações no Streamlit Cloud**

- **FFmpeg**: Não pode ser instalado automaticamente
- **Alternativas**: A aplicação usa MoviePy ou ffmpeg-python como fallback
- **Funcionamento**: Quando FFmpeg falha, automaticamente tenta métodos alternativos
- **Recomendação**: Use arquivos de áudio (.wav, .mp3) diretamente para melhor compatibilidade
- **Vídeos**: A aplicação tentará processar, mas pode ser mais lentode vídeos, transcrição de áudio, teste de velocidade de internet, remoção de fundo de imagens e conversão texto-para-fala.

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

#### **Opção 1: Script Batch (Recomendado para Windows)**

```batch
run_app.bat
```

Clique duas vezes no arquivo `run_app.bat` ou execute no terminal.

#### **Opção 2: PowerShell**

```powershell
.\.venv\Scripts\python.exe -m streamlit run youtube.py
```

#### **Opção 3: Linha de Comando**

```bash
python -m streamlit run youtube.py
```

A aplicação abrirá no navegador em `http://localhost:8501`

## ☁️ **Deploy no Streamlit Cloud**

A aplicação é compatível com o Streamlit Cloud! Para deploy:

### **1. Preparar para Deploy**

Certifique-se de que o `requirements.txt` inclui todas as dependências necessárias:

```txt
# Dependências essenciais para Streamlit Cloud
streamlit>=1.26.0
yt-dlp>=2024.0.0
SpeechRecognition>=3.10.0
pydub>=0.25.1
fpdf>=1.7.2
Pillow>=10.0.0

# Para compatibilidade com Streamlit Cloud (extração de áudio alternativa)
moviepy>=1.0.3
ffmpeg-python>=0.2.0
```

### **2. Limitações no Streamlit Cloud**

- **FFmpeg**: Não pode ser instalado automaticamente
- **Alternativas**: A aplicação usa MoviePy ou ffmpeg-python como fallback
- **Recomendação**: Use arquivos de áudio (.wav, .mp3) diretamente para melhor compatibilidade
- **Vídeos**: Converta para áudio localmente antes do upload

### **3. Dicas para Streamlit Cloud**

- **Upload de arquivos**: Limite de 200MB por arquivo
- **Processamento**: Pode ser mais lento devido aos recursos compartilhados
- **Transcrição**: Funciona melhor com arquivos de áudio puros
- **Armazenamento**: Arquivos temporários são removidos automaticamente

### **4. Deploy Steps**

1. Faça upload do código para GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório GitHub
4. Selecione o arquivo principal (`youtube.py`)
5. Clique em "Deploy"

## 🆕 **Melhorias Recentes (Outubro 2025)**

### **Compatibilidade com Streamlit Cloud**

- **Detecção automática**: Sistema identifica automaticamente quando está rodando no Streamlit Cloud
- **Fallback inteligente**: Quando FFmpeg não pode ser instalado, usa MoviePy ou ffmpeg-python
- **Recuperação de falhas**: Se instalação do FFmpeg falha, automaticamente tenta métodos alternativos
- **Mensagens claras**: Orientação específica para usuários sobre uso de arquivos de áudio vs vídeo

### **Dependências Opcionais**

- **MoviePy**: Para extração de áudio alternativa
- **ffmpeg-python**: Biblioteca Python para processamento de vídeo/audio
- **Instalação automática**: Bibliotecas incluídas no `requirements.txt` para deploy fácil

## 🔧 Solução de Problemas

### **Problemas no Streamlit Cloud**

#### **"FFmpeg não encontrado" no Streamlit Cloud**

- **Causa**: O Streamlit Cloud não permite instalação de pacotes do sistema
- **Solução**:
  - Use arquivos de áudio (.wav, .mp3) diretamente
  - Converta vídeos para áudio localmente antes do upload
  - A aplicação tentará usar MoviePy ou ffmpeg-python como alternativa

#### **Extração de áudio falha no Streamlit Cloud**

- **Sintomas**: "Falha ao instalar FFmpeg automaticamente"
- **Solução**:
  - Use arquivos de áudio puros em vez de vídeos
  - Converta vídeos localmente: `ffmpeg -i video.mp4 audio.wav`
  - Upload apenas o arquivo de áudio resultante

#### **Processamento lento no Streamlit Cloud**

- **Causa**: Recursos compartilhados e limitações de CPU/memória
- **Soluções**:
  - Use arquivos menores (< 50MB)
  - Prefira áudio em formato WAV ou MP3
  - Evite vídeos longos para transcrição

### **Problemas Gerais**

- Instale FFmpeg e adicione ao PATH
- No Windows: Reinicie o terminal após adicionar ao PATH

### **Erro: "Module not found"**

- Certifique-se de que instalou todas as dependências: `pip install -r requirements.txt`
- Verifique se está no ambiente virtual correto

### **Erro: "Fatal error in launcher: Unable to create process"**

- Use o script `run_app.bat` (recomendado)
- Ou execute: `.\.venv\Scripts\python.exe -m streamlit run youtube.py`
- Evite usar `streamlit run youtube.py` diretamente no PowerShell

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
