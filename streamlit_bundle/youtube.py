import streamlit as st
import yt_dlp
import os
import re
import glob
import speech_recognition as sr
from pydub import AudioSegment
from fpdf import FPDF
import time
from PIL import Image, ImageEnhance
from yt_dlp.utils import DownloadError

# Importação opcional para speedtest
try:
    import speedtest
    SPEEDTEST_AVAILABLE = True
except ImportError:
    SPEEDTEST_AVAILABLE = False
    speedtest = None

# Importação opcional para remoção de fundo
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# Import opcional para desenho/máscara interativa (streamlit-drawable-canvas)
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# Optional TTS/backends availability flags (set once at import time)
try:
    import edge_tts  # type: ignore
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

try:
    from gtts import gTTS  # type: ignore
    GTTTS_AVAILABLE = True
except Exception:
    GTTTS_AVAILABLE = False

try:
    import pyttsx3  # type: ignore
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False


# HTML5 fallback canvas component using st.components.v1.html (module-level so it's callable everywhere)
def html5_canvas_component(data_url, width, height):
    """Embed a small HTML5 canvas that displays data_url as background and
    returns a PNG data-url representing the drawn mask when the user clicks Save.
    """
    try:
        from streamlit.components.v1 import html as components_html
    except Exception as e:
        st.error(f"Não foi possível importar st.components.v1: {e}")
        return None

    # Sanitize dimensions
    try:
        w = int(width)
        h = int(height)
    except Exception:
        w, h = 640, 480


# Inicializar estado da sessão
def init_session_state():
    if 'downloads' not in st.session_state:
        st.session_state.downloads = []

def add_download_to_session(download_data):
    if 'downloads' not in st.session_state:
        st.session_state.downloads = []
    st.session_state.downloads.append(download_data)

def show_downloads_list():
    if 'downloads' in st.session_state and st.session_state.downloads:
        st.subheader("📥 Downloads Prontos")
        for i, download in enumerate(st.session_state.downloads):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{download['title']}** ({download['extension'].upper()})")
            with col2:
                st.download_button(
                    label="📥 Baixar",
                    data=download['data'],
                    file_name=download['filename'],
                    mime=f"{'video' if download['extension'] == 'mp4' else 'audio'}/{download['extension']}",
                    key=f"download_{i}_{hash(download['filename'])}"
                )
        
        if st.button("🗑️ Limpar Lista de Downloads"):
            st.session_state.downloads = []
            st.rerun()

def download_video(url, audio_only=False):
    """Baixa vídeo ou áudio do YouTube com múltiplos fallbacks de formato.

    - Trata problemas recentes de nsig / SABR forçado.
    - Faz várias tentativas com diferentes strings de formato.
    - Retorna dicionário compatível com o restante da aplicação ou None.
    """

    def sanitize_filename(title: str) -> str:
        title = re.sub(r'[^\w\s\-_.]', '', title)
        return title.strip()

    # Lista de tentativas para vídeo completo
    video_format_attempts = [
        # Tenta vídeo+áudio melhor combinação explícita
        {
            'format': 'bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/bv/best',
            'merge_output_format': 'mp4'
        },
        # Fallback geral
        {
            'format': 'best',
            'merge_output_format': 'mp4'
        },
    ]
    # Config base comum
    base_opts = {
        'outtmpl': 'temp_%(title)s.%(ext)s',
        'restrictfilenames': True,
        'ignoreerrors': True,
        'noplaylist': True,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'cachedir': False,
        'quiet': True,
        'retries': 3,
    }

    if audio_only:
        attempts = [
            {
                **base_opts,
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192'
                }]
            }
        ]
        target_extension = 'mp3'
    else:
        attempts = [ { **base_opts, **fmt } for fmt in video_format_attempts ]
        target_extension = 'mp4'

    last_error = None

    for attempt_index, ydl_opts in enumerate(attempts, start=1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    last_error = "Informações do vídeo não retornadas."
                    continue

                # Em playlists, yt-dlp pode retornar uma lista
                if 'entries' in info and isinstance(info['entries'], list):
                    info = next((e for e in info['entries'] if e), None)
                    if not info:
                        last_error = "Entrada da playlist vazia."
                        continue

                title = info.get('title', 'video')
                sanitized_title = sanitize_filename(title)

                # Localiza arquivo recém-criado
                candidate_patterns = [
                    f"temp_{title}.{target_extension}",
                    f"temp_{sanitized_title}.{target_extension}",
                    f"*{title[:25]}*.{target_extension}",
                ]

                downloaded_file = None
                for pattern in candidate_patterns:
                    files = glob.glob(pattern)
                    if files:
                        downloaded_file = max(files, key=os.path.getctime)
                        break

                if not downloaded_file:
                    # Busca qualquer arquivo do tipo certo gerado por esta execução
                    all_files = glob.glob(f"*.{target_extension}")
                    if all_files:
                        downloaded_file = max(all_files, key=os.path.getctime)

                if downloaded_file and os.path.exists(downloaded_file):
                    with open(downloaded_file, 'rb') as f:
                        file_data = f.read()
                    os.remove(downloaded_file)
                    return {
                        'data': file_data,
                        'filename': f"{sanitized_title}.{target_extension}",
                        'title': title,
                        'extension': target_extension
                    }
                else:
                    last_error = f"Arquivo não encontrado após tentativa {attempt_index}."
        except DownloadError as de:
            last_error = f"DownloadError tentativa {attempt_index}: {de}"
        except Exception as e:
            # Captura erros gerais (inclui possíveis falhas de nsig)
            err_text = str(e)
            if 'nsig' in err_text.lower():
                st.warning("⚠️ Possível problema de 'nsig' do YouTube. Atualize o yt-dlp: pip install -U yt-dlp")
            last_error = f"Erro tentativa {attempt_index}: {err_text}"

    if last_error:
        st.error(f"Erro durante o download: {last_error}")
    return None

def list_available_formats(url: str):
    """Função utilitária para listar os formatos disponíveis (debug)."""
    ydl_opts = { 'skip_download': True, 'quiet': True }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            formats = info.get('formats', [])
            rows = []
            for f in formats:
                if not f.get('url'):
                    continue
                rows.append(
                    f"{f.get('format_id')}: {f.get('ext')} {f.get('resolution') or ''} {f.get('fps') or ''} {f.get('abr') or ''}".strip()
                )
            return rows[:50]
    except Exception as e:
        return [f"Erro ao listar formatos: {e}"]

# ---------------------- Funções de transcrição de áudio ---------------------- #
import uuid
import subprocess
import tempfile
from datetime import datetime

def init_transcription_session_state():
    """Inicializa estado da sessão para transcrições."""
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    if 'transcription_cache' not in st.session_state:
        st.session_state.transcription_cache = {}

def add_transcription_to_session(transcription_data):
    """Adiciona transcrição à lista de sessão."""
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    st.session_state.transcriptions.append(transcription_data)

def show_transcriptions_list():
    """Mostra lista de transcrições disponíveis com informações detalhadas."""
    if 'transcriptions' in st.session_state and st.session_state.transcriptions:
        st.subheader("📝 Transcrições Concluídas")
        
        for i, transcription in enumerate(st.session_state.transcriptions):
            # Cabeçalho com informações resumidas
            success_rate = transcription.get('success_rate', 0)
            status_emoji = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 50 else "❌"
            
            with st.expander(f"{status_emoji} {transcription['title']} - {transcription['timestamp']} (🎬 {transcription.get('video_duration', 'N/A')} | {success_rate:.0f}% sucesso)"):
                # Informações detalhadas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**⏰ Tempos:**")
                    st.write(f"Início: {transcription.get('start_time', 'N/A')}")
                    st.write(f"Término: {transcription.get('end_time', 'N/A')}")
                    st.write(f"Duração: {transcription['execution_time']:.1f}s")
                
                with col2:
                    st.write("**📊 Estatísticas:**")
                    st.write(f"Idioma: {transcription.get('language', 'N/A')}")
                    st.write(f"Segmentos: {transcription.get('chunks_count', 0)}")
                    st.write(f"Taxa de sucesso: {success_rate:.1f}%")
                
                with col3:
                    st.write("**📥 Downloads:**")
                    # Botão PDF
                    st.download_button(
                        label="📥 Baixar PDF",
                        data=transcription['pdf_data'],
                        file_name=f"transcricao_{transcription['title'][:30].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{i}"
                    )
                    # Botão TXT
                    st.download_button(
                        label="📥 Baixar TXT",
                        data=transcription['text'].encode('utf-8'),
                        file_name=f"transcricao_{transcription['title'][:30].replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"txt_{i}"
                    )
                
                # Prévia do conteúdo
                st.write("**📄 Prévia do Conteúdo:**")
                preview_text = transcription['text'][:500] + "..." if len(transcription['text']) > 500 else transcription['text']
                # Use a non-empty label but hide it to avoid Streamlit accessibility warnings
                st.text_area("Pré-visualização da transcrição", preview_text, height=150, key=f"preview_{i}", label_visibility='collapsed')
        
        # Botão para limpar todas as transcrições
        if st.button("🗑️ Limpar Todas as Transcrições", type="secondary"):
            st.session_state.transcriptions = []
            st.rerun()
    else:
        st.info("📝 Nenhuma transcrição salva ainda. Faça sua primeira transcrição!")

def safe_remove_file(filepath):
    """Remove arquivo de forma segura, ignorando erros."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except:
        pass

def extract_audio_with_ffmpeg(input_path, output_path, quality='standard'):
    """Extrai áudio usando subprocess para melhor controle com configurações de qualidade."""
    try:
        # Remove arquivo de saída se existir
        safe_remove_file(output_path)
        
        # Configurações baseadas na qualidade
        if quality == 'high':
            sample_rate = '44100'
            channels = '2'  # Stereo
        else:
            sample_rate = '16000'  # Otimizado para speech recognition
            channels = '1'  # Mono
        
        cmd = [
            'ffmpeg', '-y',  # -y para sobrescrever sem perguntar
            '-i', input_path,
            '-vn',  # Sem vídeo
            '-acodec', 'pcm_s16le',
            '-ar', sample_rate,
            '-ac', channels,
            '-af', 'volume=2.0',  # Aumentar volume
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Erro FFmpeg: {result.stderr}")
            return False
        
        return os.path.exists(output_path)
    except Exception as e:
        st.error(f"Erro ao extrair áudio: {e}")
        return False

def split_audio(audio_path, chunk_length_ms=120000):  # 2 minutos por chunk
    """Divide áudio em chunks menores."""
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunks.append(chunk)
        return chunks
    except Exception as e:
        st.error(f"Erro ao dividir áudio: {e}")
        return []

def transcribe_audio_chunk(audio_chunk, recognizer, chunk_index, language='auto'):
    """Transcreve um chunk de áudio com múltiplas tentativas e idiomas."""
    temp_file = None
    try:
        # Criar arquivo temporário único
        temp_file = f"temp_chunk_{chunk_index}_{uuid.uuid4().hex[:8]}.wav"
        audio_chunk.export(temp_file, format="wav")
        
        with sr.AudioFile(temp_file) as source:
            # Ajustar para ruído ambiente
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        
        # Lista de idiomas para tentar (se auto, tenta vários)
        if language == 'auto':
            languages_to_try = ['en-US', 'pt-BR', 'es-ES', 'fr-FR']
        else:
            languages_to_try = [language]
        
        # Tentar diferentes engines e idiomas
        for lang in languages_to_try:
            try:
                # Primeira tentativa: Google Speech Recognition
                text = recognizer.recognize_google(audio_data, language=lang)
                if text.strip():  # Se conseguiu algum texto
                    return f"[{lang}] {text}"
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                continue
        
        # Se Google falhou, tentar Whisper se disponível
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(temp_file)
            if result["text"].strip():
                return f"[Whisper] {result['text']}"
        except ImportError:
            pass
        except Exception:
            pass
        
        # Última tentativa: Sphinx (offline)
        try:
            text = recognizer.recognize_sphinx(audio_data)
            if text.strip():
                return f"[Sphinx] {text}"
        except:
            pass
                
        return f"[Chunk {chunk_index + 1}: Áudio não compreensível - tente outro idioma]"
                
    except Exception as e:
        return f"[Chunk {chunk_index + 1}: Erro - {str(e)}]"
    finally:
        safe_remove_file(temp_file)

def create_pdf_from_text(text, title, execution_time, start_time, end_time, language, chunks_stats):
    """Cria PDF a partir do texto transcrito com timestamps detalhados incluindo posição no vídeo."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Título principal
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 15, "Transcrição de Áudio/Vídeo", ln=True, align='C')
        pdf.ln(5)
        
        # Informações detalhadas
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Título: {title}", ln=True)
        
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Data da transcrição: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}", ln=True)
        pdf.cell(0, 6, f"Início do processamento: {start_time.strftime('%d/%m/%Y às %H:%M:%S')}", ln=True)
        pdf.cell(0, 6, f"Término do processamento: {end_time.strftime('%d/%m/%Y às %H:%M:%S')}", ln=True)
        pdf.cell(0, 6, f"Tempo total de processamento: {execution_time:.2f} segundos", ln=True)
        pdf.cell(0, 6, f"Duração total do vídeo/áudio: {chunks_stats.get('total_duration', 'N/A')}", ln=True)
        pdf.cell(0, 6, f"Idioma detectado/configurado: {language}", ln=True)
        pdf.cell(0, 6, f"Segmentos processados: {chunks_stats['total']} total, {chunks_stats['successful']} com sucesso", ln=True)
        pdf.cell(0, 6, f"Taxa de sucesso: {chunks_stats['success_rate']:.1f}%", ln=True)
        pdf.ln(8)
        
        # Separador
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "=" * 60, ln=True, align='C')
        pdf.cell(0, 8, "CONTEÚDO DA TRANSCRIÇÃO COM TIMESTAMPS", ln=True, align='C')
        pdf.cell(0, 8, "=" * 60, ln=True, align='C')
        pdf.ln(5)
        
        # Conteúdo principal
        pdf.set_font("Arial", size=11)
        
        # Parse segments robustly: each segment starts with a header like
        # [Segmento 1/3 | Vídeo: 00:00-02:00 | Processado em 55.0s]
        import re
        segment_pattern = re.compile(r"(\[Segmento\s+\d+/\d+\s*\|\s*Vídeo:\s*[^\]]+\])", flags=re.IGNORECASE)

        parts = segment_pattern.split(text)
        # parts will be like ['', header1, body1, header2, body2, ...]
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            body = parts[i+1].strip() if i+1 < len(parts) else ''

            # Render header
            pdf.set_font("Arial", 'B', 10)
            pdf.set_fill_color(240, 240, 240)
            safe_header = header.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 8, safe_header, ln=True, fill=True)
            pdf.set_font("Arial", size=11)
            pdf.ln(2)

            # Render body with wrapping
            if body:
                # Normalize newlines
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
                for para in paragraphs:
                    words = para.split(' ')
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > 90:
                            if current_line:
                                safe_text = current_line.encode('latin-1', 'replace').decode('latin-1')
                                pdf.multi_cell(0, 6, safe_text)
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line:
                        safe_text = current_line.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 6, safe_text)
                    pdf.ln(3)
        
        # Rodapé
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, f"Documento gerado automaticamente em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}", ln=True, align='C')
        pdf.cell(0, 5, "Timestamps indicam posição no vídeo original e tempo de processamento", ln=True, align='C')
        
        # Salvar em buffer
        from io import BytesIO
        pdf_buffer = BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin1')
        pdf_buffer.write(pdf_output)
        pdf_buffer.seek(0)
        
        return pdf_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Erro ao criar PDF: {e}")
        return None

def download_audio_from_youtube(url):
    """Baixa apenas áudio de um vídeo do YouTube temporariamente."""
    temp_file = f"temp_audio_{uuid.uuid4().hex[:8]}"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{temp_file}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'keepvideo': False,  # Não manter vídeo
        'writethumbnail': False,  # Não baixar thumbnail
        'writeinfojson': False,  # Não salvar metadata
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'video')
            
            # Procurar arquivo gerado (temporário)
            audio_file = f"{temp_file}.wav"
            if os.path.exists(audio_file):
                return audio_file, title
            
            # Tentar outros formatos se wav não foi criado
            for ext in ['wav', 'mp3', 'm4a']:
                test_file = f"{temp_file}.{ext}"
                if os.path.exists(test_file):
                    return test_file, title
                    
        return None, None
        
    except Exception as e:
        st.error(f"Erro ao baixar áudio temporário: {e}")
        return None, None

def process_transcription(source_type, source_data, source_title="audio", language='auto', audio_quality='standard'):
    """Processa transcrição de diferentes tipos de fonte com timestamps detalhados."""
    # Registrar início do processamento
    start_time = datetime.now()
    process_start = time.time()
    temp_files = []
    
    st.info(f"🕐 Início do processamento: {start_time.strftime('%d/%m/%Y às %H:%M:%S')}")
    
    try:
        # Preparar arquivo de áudio
        if source_type == "youtube":
            st.info("🔄 Baixando áudio temporário do YouTube...")
            audio_file, title = download_audio_from_youtube(source_data)
            if not audio_file:
                st.error("❌ Falha ao baixar áudio do YouTube")
                return False
            temp_files.append(audio_file)
            
        elif source_type == "upload":
            st.info("🔄 Processando arquivo enviado...")
            # Salvar arquivo temporário
            temp_input = f"temp_input_{uuid.uuid4().hex[:8]}.{source_data.name.split('.')[-1]}"
            with open(temp_input, "wb") as f:
                f.write(source_data.getbuffer())
            temp_files.append(temp_input)
            
            # Extrair áudio se necessário (temporário)
            audio_file = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
            if extract_audio_with_ffmpeg(temp_input, audio_file, audio_quality):
                temp_files.append(audio_file)
                title = source_title
            else:
                st.error("❌ Falha ao extrair áudio")
                return False
        else:
            st.error("❌ Tipo de fonte inválido")
            return False
        
        # Dividir áudio em chunks
        st.info("🔄 Dividindo áudio em segmentos...")
        chunk_size = 90000 if audio_quality == 'high' else 120000  # 1.5min para alta qualidade
        chunks = split_audio(audio_file, chunk_length_ms=chunk_size)
        
        if not chunks:
            st.error("❌ Falha ao dividir áudio")
            return False
        
        st.info(f"🔄 Transcrevendo {len(chunks)} segmentos...")
        
        # Transcrever chunks
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300  # Ajustar sensibilidade
        transcriptions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        transcription_start = time.time()
        
        # Calcular timestamps do vídeo baseado no tamanho dos chunks
        video_position = 0  # Posição atual no vídeo em milissegundos
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            # Calcular timestamps do vídeo para este segmento
            video_start_ms = video_position
            video_end_ms = video_position + len(chunk)
            video_position = video_end_ms
            
            # Converter para formato mm:ss
            def ms_to_time(ms):
                total_seconds = ms // 1000
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                return f"{minutes:02d}:{seconds:02d}"
            
            video_start_time = ms_to_time(video_start_ms)
            video_end_time = ms_to_time(video_end_ms)
            
            status_text.text(f"🔄 Segmento {i+1}/{len(chunks)} [{video_start_time}-{video_end_time}] - {datetime.now().strftime('%H:%M:%S')}")
            progress_bar.progress((i + 1) / len(chunks))
            
            text = transcribe_audio_chunk(chunk, recognizer, i, language)
            chunk_end = time.time()
            
            # Adicionar timestamp completo (processamento + vídeo)
            chunk_duration = chunk_end - chunk_start
            timestamp_info = f"[Segmento {i+1}/{len(chunks)} | Vídeo: {video_start_time}-{video_end_time} | Processado em {chunk_duration:.1f}s]"
            transcriptions.append(f"{timestamp_info}\n{text}")
            
        transcription_end = time.time()
        
        # Compilar resultado final
        full_text = "\n\n".join(transcriptions)
        
        # Registrar fim do processamento
        end_time = datetime.now()
        execution_time = time.time() - process_start
        
        st.info(f"🕐 Término do processamento: {end_time.strftime('%d/%m/%Y às %H:%M:%S')}")
        
        # Estatísticas
        successful_chunks = sum(1 for t in transcriptions if not 'não compreensível' in t)
        chunks_stats = {
            'total': len(transcriptions),
            'successful': successful_chunks,
            'success_rate': (successful_chunks / len(transcriptions)) * 100 if transcriptions else 0,
            'total_duration': ms_to_time(video_position)
        }
        
        # Criar PDF com timestamps
        pdf_data = create_pdf_from_text(full_text, title, execution_time, start_time, end_time, language, chunks_stats)
        
        if pdf_data:
            # Adicionar à sessão (sem salvar arquivos no disco)
            transcription_data = {
                'title': title,
                'text': full_text,
                'pdf_data': pdf_data,
                'timestamp': start_time.strftime('%d/%m/%Y %H:%M'),
                'start_time': start_time.strftime('%d/%m/%Y às %H:%M:%S'),
                'end_time': end_time.strftime('%d/%m/%Y às %H:%M:%S'),
                'execution_time': execution_time,
                'language': language,
                'chunks_count': len(transcriptions),
                'success_rate': chunks_stats['success_rate'],
                'video_duration': chunks_stats['total_duration']
            }
            
            add_transcription_to_session(transcription_data)
            
            st.success(f"✅ Transcrição concluída!")
            st.success(f"⏱️ Tempo total: {execution_time:.1f} segundos")
            st.success(f"🎬 Duração do vídeo: {chunks_stats['total_duration']}")
            st.success(f"📝 {successful_chunks}/{len(transcriptions)} segmentos transcritos com sucesso ({chunks_stats['success_rate']:.1f}%)")
            
            # Informações detalhadas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Início", start_time.strftime('%H:%M:%S'))
            with col2:
                st.metric("Término", end_time.strftime('%H:%M:%S'))
            with col3:
                st.metric("Processamento", f"{execution_time:.1f}s")
            with col4:
                st.metric("Duração Vídeo", chunks_stats['total_duration'])
            
            return True
        else:
            st.error("❌ Erro ao gerar PDF")
            return False
            
    except Exception as e:
        end_time = datetime.now()
        st.error(f"❌ Erro durante transcrição: {e}")
        st.error(f"🕐 Falha em: {end_time.strftime('%d/%m/%Y às %H:%M:%S')}")
        return False
        
    finally:
        # Limpar TODOS os arquivos temporários automaticamente
        st.info("🧹 Limpando arquivos temporários...")
        for temp_file in temp_files:
            safe_remove_file(temp_file)
        
        # Limpar qualquer arquivo temp_ restante no diretório
        import glob
        for pattern in ["temp_*", "*.tmp"]:
            for file in glob.glob(pattern):
                safe_remove_file(file)

# ---------------------- Função SpeedTest ---------------------- #
def run_speedtest():
    st.header('🚀 Teste de Velocidade da Internet')

    if not SPEEDTEST_AVAILABLE:
        st.error("❌ Módulo speedtest não disponível. Instale as dependências necessárias.")
        st.code("pip install speedtest-cli")
        return

    st.write('Clique no botão abaixo para iniciar o teste de velocidade.')

    # Quick mode: allow user to skip/short-circuit slow server discovery
    if 'speedtest_quick_mode' not in st.session_state:
        st.session_state['speedtest_quick_mode'] = False
    quick_mode = st.checkbox('⚡ Modo rápido (pulsa descoberta de servidor para iniciar mais rápido)', value=st.session_state.get('speedtest_quick_mode', False))

    # Ensure session history is loaded from disk
    try:
        import json as _json, pathlib as _pathlib
        hist_path = _pathlib.Path(os.path.join(os.getcwd(), 'speedtest_history.json'))
        if 'speedtest_history' not in st.session_state:
            if hist_path.exists():
                try:
                    with hist_path.open('r', encoding='utf-8') as _f:
                        st.session_state['speedtest_history'] = _json.load(_f)
                except Exception:
                    st.session_state['speedtest_history'] = []
            else:
                st.session_state['speedtest_history'] = []
    except Exception:
        if 'speedtest_history' not in st.session_state:
            st.session_state['speedtest_history'] = []

    # Clear history button
    cols_control = st.columns([1, 4])
    with cols_control[0]:
        if st.button('🗑️ Limpar histórico'):
            st.session_state['speedtest_history'] = []
            try:
                import pathlib as _pathlib
                hist_path = _pathlib.Path(os.path.join(os.getcwd(), 'speedtest_history.json'))
                if hist_path.exists():
                    hist_path.unlink()
            except Exception:
                pass

    if st.button('📡 Iniciar Teste de Velocidade'):
        # persist quick-mode selection into session so retry buttons can toggle it
        st.session_state['speedtest_quick_mode'] = bool(quick_mode)
        with st.spinner('Testando a velocidade da sua internet... Isso pode levar alguns minutos'):
            try:
                st.info('⏳ Iniciando teste de velocidade...')
                s = speedtest.Speedtest()

                # Try to discover best server but timebox when in quick mode to avoid long waits
                if quick_mode:
                    try:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                            fut = _ex.submit(s.get_best_server)
                            try:
                                fut.result(timeout=1.5)
                            except concurrent.futures.TimeoutError:
                                st.info('Modo rápido: descoberta do melhor servidor demorando — continuando sem bloqueio.')
                            except Exception:
                                # non-fatal, continue anyway
                                pass
                    except Exception:
                        # If threadpool not available for any reason, fall back to non-blocking attempt
                        try:
                            s.get_best_server()
                        except Exception:
                            pass
                else:
                    try:
                        s.get_best_server()
                    except Exception:
                        # non-fatal: proceed to tests which will pick a server when needed
                        pass


                # placeholders and live chart data
                chart_ph = st.empty()
                c1, c2, c3 = st.columns(3)
                p1 = c1.empty(); p2 = c2.empty(); p3 = c3.empty()

                # Use a small in-memory list to collect progressive samples for a simple live chart
                samples = []

                def _push_sample(stage, value):
                    # stage: 'download' or 'upload' or 'ping'
                    samples.append({'stage': stage, 'value': float(value), 'ts': datetime.now()})
                    try:
                        import pandas as _pd
                        df = _pd.DataFrame(samples)
                        # pivot to wide format so lines are separate series
                        if not df.empty:
                            try:
                                import altair as alt
                                chart = alt.Chart(df).mark_line(point=True).encode(
                                    x='ts:T',
                                    y=alt.Y('value:Q', title='Mbps / ms'),
                                    color='stage:N'
                                ).properties(height=300)
                                chart_ph.altair_chart(chart, use_container_width=True)
                            except Exception:
                                # streamlit fallback
                                try:
                                    pivot = df.pivot_table(index='ts', columns='stage', values='value')
                                    chart_ph.line_chart(pivot)
                                except Exception:
                                    chart_ph.write(df)
                    except Exception:
                        # best effort
                        chart_ph.write(f"{stage}: {value}")

                # ------------------ Download phase (show chart while running) ------------------
                st.info('🔍 Testando download...')
                chart_ph = chart_ph or st.empty()
                p1.metric('📥 Download', '—')
                import concurrent.futures, time

                def _do_download():
                    return s.download() / 1_000_000

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_do_download)
                    fake = 0.0
                    # animate a temporary chart while the blocking download runs
                    while not fut.done():
                        # simple easing growth to give perceptible motion
                        fake = fake + max(1.5, fake * 0.08)
                        if fake > 2000:
                            fake = 2000
                        try:
                            # Show a small donut that updates during the download phase
                            import plotly.graph_objects as _go
                            try:
                                up_val = float(upload_speed)
                            except Exception:
                                up_val = 0.0001
                            # Render as a half-donut (semicircle) and clamp values to 0..1000
                            MAX_SCALE = 1000.0
                            try:
                                val = float(fake)
                            except Exception:
                                val = 0.0
                            val_clamped = max(0.0, min(MAX_SCALE, val))
                            remaining = max(0.0001, MAX_SCALE - val_clamped)
                            # primary color for value, subtle grey for remainder
                            donut = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                            labels=[f'Download (Mbps)', ''],
                                                            hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                            domain={'x':[0,1], 'y':[0,0.5]},
                                                            marker=dict(colors=['#4cc9f0', '#e9ecef']))])
                            donut.update_traces(textinfo='none', hoverinfo='label+value')
                            donut.update_layout(height=260, margin={'t':6,'b':6,'l':6,'r':6}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                               annotations=[dict(text=f"{val_clamped:.1f} Mbps", x=0.5, y=0.18, font=dict(size=14), showarrow=False)])
                            chart_ph.plotly_chart(donut, use_container_width=True)
                            try:
                                # Also show a compact semicircle in the small Download column while measuring
                                compact = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                                 labels=[f'DL', ''],
                                                                 hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                                 domain={'x':[0,1], 'y':[0,0.6]},
                                                                 marker=dict(colors=['#4cc9f0', '#e9ecef']))])
                                compact.update_traces(textinfo='none', hoverinfo='label+value')
                                compact.update_layout(height=120, margin={'t':2,'b':2,'l':2,'r':2}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                                      annotations=[dict(text=f"{val_clamped:.1f}", x=0.5, y=0.12, font=dict(size=12), showarrow=False)])
                                p1.plotly_chart(compact, use_container_width=True)
                            except Exception:
                                try:
                                    p1.metric('📥 Download', f"{val_clamped:.1f} Mbps")
                                except Exception:
                                    pass
                        except Exception:
                            chart_ph.write(f'⌛ Download em andamento: {fake:.1f} Mbps')
                        time.sleep(0.12)

                    # get actual result
                    # Capture exceptions from the download thread explicitly so we can
                    # provide tailored guidance for common HTTP errors (403 Forbidden).
                    try:
                        download_speed = float(fut.result())
                    except Exception as _e:
                        # Re-raise to be caught by outer handler below with context
                        raise _e
                    # final render for download
                    try:
                        # Final per-phase donut showing the actual download value
                        import plotly.graph_objects as _go
                        try:
                            up_val = float(upload_speed)
                        except Exception:
                            up_val = 0.0001
                        # Final per-phase half-donut for Download, clamped to 0..1000
                        MAX_SCALE = 1000.0
                        try:
                            val = float(download_speed)
                        except Exception:
                            val = 0.0
                        val_clamped = max(0.0, min(MAX_SCALE, val))
                        remaining = max(0.0001, MAX_SCALE - val_clamped)
                        donut = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                        labels=[f'Download (Mbps)', ''],
                                                        hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                        domain={'x':[0,1], 'y':[0,0.5]},
                                                        marker=dict(colors=['#4cc9f0', '#e9ecef']))])
                        donut.update_traces(textinfo='none', hoverinfo='label+value')
                        donut.update_layout(height=260, margin={'t':6,'b':6,'l':6,'r':6}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                           annotations=[dict(text=f"{val_clamped:.2f} Mbps", x=0.5, y=0.18, font=dict(size=14), showarrow=False)])
                        chart_ph.plotly_chart(donut, use_container_width=True)
                        try:
                            # final small column chart for Download
                            compact = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                             labels=[f'DL', ''],
                                                             hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                             domain={'x':[0,1], 'y':[0,0.6]},
                                                             marker=dict(colors=['#4cc9f0', '#e9ecef']))])
                            compact.update_traces(textinfo='none', hoverinfo='label+value')
                            compact.update_layout(height=120, margin={'t':2,'b':2,'l':2,'r':2}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                                  annotations=[dict(text=f"{val_clamped:.2f}", x=0.5, y=0.12, font=dict(size=12), showarrow=False)])
                            p1.plotly_chart(compact, use_container_width=True)
                        except Exception:
                            try:
                                p1.metric('📥 Download', f'{download_speed:.2f} Mbps')
                            except Exception:
                                pass
                    except Exception:
                        chart_ph.write(f'Download: {download_speed:.2f} Mbps')

                    p1.metric('📥 Download', f'{download_speed:.2f} Mbps')
                    time.sleep(0.6)
                    chart_ph.empty()

                # ------------------ Upload phase (show chart while running) ------------------
                st.info('🔍 Testando upload...')
                chart_ph = st.empty()
                p2.metric('📤 Upload', '—')

                def _do_upload():
                    return s.upload() / 1_000_000

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex2:
                    fut2 = ex2.submit(_do_upload)
                    fake2 = 0.0
                    while not fut2.done():
                        fake2 = fake2 + max(1.0, fake2 * 0.06)
                        if fake2 > 2000:
                            fake2 = 2000
                        try:
                            # Show a donut during the upload phase
                            import plotly.graph_objects as _go
                            try:
                                dl_val = float(download_speed)
                            except Exception:
                                dl_val = 0.0001
                            # Upload live: half-donut showing upload progress vs 1000 Mbps
                            MAX_SCALE = 1000.0
                            try:
                                val = float(fake2)
                            except Exception:
                                val = 0.0
                            val_clamped = max(0.0, min(MAX_SCALE, val))
                            remaining = max(0.0001, MAX_SCALE - val_clamped)
                            donut = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                            labels=[f'Upload (Mbps)', ''],
                                                            hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                            domain={'x':[0,1], 'y':[0,0.5]},
                                                            marker=dict(colors=['#f72585', '#e9ecef']))])
                            donut.update_traces(textinfo='none', hoverinfo='label+value')
                            donut.update_layout(height=260, margin={'t':6,'b':6,'l':6,'r':6}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                               annotations=[dict(text=f"{val_clamped:.1f} Mbps", x=0.5, y=0.18, font=dict(size=14), showarrow=False)])
                            chart_ph.plotly_chart(donut, use_container_width=True)
                            try:
                                # Also show a compact semicircle in the small Upload column while measuring
                                compact_u = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                                     labels=[f'UL', ''],
                                                                     hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                                     domain={'x':[0,1], 'y':[0,0.6]},
                                                                     marker=dict(colors=['#f72585', '#e9ecef']))])
                                compact_u.update_traces(textinfo='none', hoverinfo='label+value')
                                compact_u.update_layout(height=120, margin={'t':2,'b':2,'l':2,'r':2}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                                        annotations=[dict(text=f"{val_clamped:.1f}", x=0.5, y=0.12, font=dict(size=12), showarrow=False)])
                                p2.plotly_chart(compact_u, use_container_width=True)
                            except Exception:
                                try:
                                    p2.metric('📤 Upload', f"{val_clamped:.1f} Mbps")
                                except Exception:
                                    pass
                        except Exception:
                            chart_ph.write(f'⌛ Upload em andamento: {fake2:.1f} Mbps')
                        time.sleep(0.12)

                    try:
                        upload_speed = float(fut2.result())
                    except Exception as _e:
                        raise _e
                    try:
                        # Final per-phase donut showing the actual upload value
                        import plotly.graph_objects as _go
                        try:
                            dl_val = float(download_speed)
                        except Exception:
                            dl_val = 0.0001
                        # Final per-phase half-donut for Upload, clamped to 0..1000
                        MAX_SCALE = 1000.0
                        try:
                            val = float(upload_speed)
                        except Exception:
                            val = 0.0
                        val_clamped = max(0.0, min(MAX_SCALE, val))
                        remaining = max(0.0001, MAX_SCALE - val_clamped)
                        donut = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                        labels=[f'Upload (Mbps)', ''],
                                                        hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                        domain={'x':[0,1], 'y':[0,0.5]},
                                                        marker=dict(colors=['#f72585', '#e9ecef']))])
                        donut.update_traces(textinfo='none', hoverinfo='label+value')
                        donut.update_layout(height=260, margin={'t':6,'b':6,'l':6,'r':6}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                           annotations=[dict(text=f"{val_clamped:.2f} Mbps", x=0.5, y=0.18, font=dict(size=14), showarrow=False)])
                        chart_ph.plotly_chart(donut, use_container_width=True)
                        try:
                            # final small column chart for Upload
                            compact_u = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                                 labels=[f'UL', ''],
                                                                 hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                                 domain={'x':[0,1], 'y':[0,0.6]},
                                                                 marker=dict(colors=['#f72585', '#e9ecef']))])
                            compact_u.update_traces(textinfo='none', hoverinfo='label+value')
                            compact_u.update_layout(height=120, margin={'t':2,'b':2,'l':2,'r':2}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                                    annotations=[dict(text=f"{val_clamped:.2f}", x=0.5, y=0.12, font=dict(size=12), showarrow=False)])
                            p2.plotly_chart(compact_u, use_container_width=True)
                        except Exception:
                            try:
                                p2.metric('📤 Upload', f'{upload_speed:.2f} Mbps')
                            except Exception:
                                pass
                    except Exception:
                        chart_ph.write(f'Upload: {upload_speed:.2f} Mbps')

                    p2.metric('📤 Upload', f'{upload_speed:.2f} Mbps')
                    time.sleep(0.6)
                    chart_ph.empty()

                # ------------------ Ping and final metrics ------------------
                # Show a brief animated semicircle in the small Ping column while results are prepared
                try:
                    import plotly.graph_objects as _go
                    fake_ping = 0.0
                    # short animated placeholder (approx 0.6s)
                    for _ in range(6):
                        fake_ping = min(300.0, fake_ping + max(1.0, fake_ping * 0.2) + 2.0)
                        MAX_SCALE = 1000.0
                        val_clamped = max(0.0, min(MAX_SCALE, float(fake_ping)))
                        remaining = max(0.0001, MAX_SCALE - val_clamped)
                        compact_p = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                            labels=['Ping (ms)', ''],
                                                            hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                            domain={'x':[0,1], 'y':[0,0.6]},
                                                            marker=dict(colors=['#ffd166', '#e9ecef']))])
                        compact_p.update_traces(textinfo='none', hoverinfo='label+value')
                        compact_p.update_layout(height=120, margin={'t':2,'b':2,'l':2,'r':2}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                                 annotations=[dict(text=f"{val_clamped:.0f} ms", x=0.5, y=0.12, font=dict(size=12), showarrow=False)])
                        p3.plotly_chart(compact_p, use_container_width=True)
                        time.sleep(0.08)
                except Exception:
                    try:
                        p3.metric('⚡ Ping', '—')
                    except Exception:
                        pass

                results = s.results.dict()
                ping_v = float(results.get('ping', 0.0)) if results else 0.0
                # Final: display ping as compact semicircle in p3
                try:
                    import plotly.graph_objects as _go
                    MAX_SCALE = 1000.0
                    val = float(ping_v)
                    val_clamped = max(0.0, min(MAX_SCALE, val))
                    remaining = max(0.0001, MAX_SCALE - val_clamped)
                    compact_p = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                        labels=['Ping (ms)', ''],
                                                        hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                        domain={'x':[0,1], 'y':[0,0.6]},
                                                        marker=dict(colors=['#ffd166', '#e9ecef']))])
                    compact_p.update_traces(textinfo='none', hoverinfo='label+value')
                    compact_p.update_layout(height=120, margin={'t':2,'b':2,'l':2,'r':2}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                             annotations=[dict(text=f"{val_clamped:.2f} ms", x=0.5, y=0.12, font=dict(size=12), showarrow=False)])
                    p3.plotly_chart(compact_p, use_container_width=True)
                except Exception:
                    try:
                        p3.metric('⚡ Ping', f'{ping_v:.2f} ms')
                    except Exception:
                        pass

                st.success('✅ Teste concluído!')

                # Final display: show speedometer-style gauges for Download and Upload
                try:
                    import plotly.graph_objects as go
                    # choose a sensible max for gauge
                    guessed_max = int(max(100, max(download_speed or 0.0, upload_speed or 0.0) * 1.5))
                    colg1, colg2, colg3 = st.columns([1,1,1])
                    # color bands helper
                    def _gauge_steps(maxv):
                        # Create 3 bands: green (0-60%), yellow (60-85%), red (85-100%)
                        return [
                            {'range':[0, maxv*0.6], 'color':'#7de29a'},
                            {'range':[maxv*0.6, maxv*0.85], 'color':'#ffd166'},
                            {'range':[maxv*0.85, maxv], 'color':'#ff6b6b'}
                        ]

                    with colg1:
                        try:
                            # Render a compact donut (rosca) chart showing Download vs Upload
                            import plotly.graph_objects as _go

                            # Compact half-donut for Download (0..1000 scale)
                            MAX_SCALE = 1000.0
                            try:
                                val = float(download_speed)
                            except Exception:
                                val = 0.0
                            val_clamped = max(0.0, min(MAX_SCALE, val))
                            remaining = max(0.0001, MAX_SCALE - val_clamped)
                            donut = _go.Figure(data=[_go.Pie(values=[val_clamped, remaining],
                                                            labels=[f'Download (Mbps)', ''],
                                                            hole=0.6, sort=False, rotation=180, direction='clockwise',
                                                            domain={'x':[0,1], 'y':[0,0.5]},
                                                            marker=dict(colors=['#4cc9f0', '#e9ecef']))])
                            donut.update_traces(textinfo='none', hoverinfo='label+value')
                            donut.update_layout(height=320, margin={'t':30,'b':10,'l':10,'r':10}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                               annotations=[dict(text=f"{val_clamped:.2f} Mbps", x=0.5, y=0.18, font=dict(size=14), showarrow=False)])
                            st.plotly_chart(donut, use_container_width=True)
                        except Exception:
                            # Fallback to a simple metric if plotly isn't installed or fails
                            st.metric('📥 Download', f"{download_speed:.2f} Mbps")
                    with colg2:
                        try:
                            steps = _gauge_steps(guessed_max)
                            fig_u = go.Figure(go.Indicator(
                                mode='gauge+number+delta',
                                value=float(upload_speed),
                                number={'suffix':' Mbps', 'font':{'size':20}},
                                title={'text':'Upload', 'font':{'size':14}},
                                delta={'reference': guessed_max/2, 'relative': True, 'increasing':{'color':'#1e7bd2'}, 'decreasing':{'color':'#c0392b'}},
                                gauge={'axis':{'range':[0, guessed_max], 'tickwidth':1, 'tickcolor':'#666'},
                                       'bar':{'color':'rgba(0,0,0,0.6)'},
                                       'steps':steps,
                                       'threshold':{'line':{'color':'#222','width':3}, 'value': guessed_max*0.9}}
                            ))
                            fig_u.update_layout(height=320, margin={'t':30,'b':10,'l':10,'r':10}, paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_u, use_container_width=True)
                        except Exception:
                            st.metric('📤 Upload', f"{upload_speed:.2f} Mbps")
                    with colg3:
                        try:
                            ping_max = max(50, int(ping_v * 3) if ping_v else 200)
                            steps_p = _gauge_steps(ping_max)
                            fig_p = go.Figure(go.Indicator(
                                mode='gauge+number',
                                value=float(ping_v),
                                number={'suffix':' ms', 'font':{'size':20}},
                                title={'text':'Ping', 'font':{'size':14}},
                                gauge={'axis':{'range':[0, ping_max], 'tickwidth':1, 'tickcolor':'#666'},
                                       'bar':{'color':'rgba(0,0,0,0.6)'},
                                       'steps':steps_p,
                                       'threshold':{'line':{'color':'#222','width':3}, 'value': ping_max*0.8}}
                            ))
                            fig_p.update_layout(height=320, margin={'t':30,'b':10,'l':10,'r':10}, paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_p, use_container_width=True)
                        except Exception:
                            st.metric('⚡ Ping', f"{ping_v:.2f} ms")
                except Exception:
                    # Outer plotly/gauge block failed (e.g., plotly not installed or rendering error).
                    # Fall back to simple metrics so UI still shows results.
                    try:
                        p1.metric('📥 Download', f"{download_speed:.2f} Mbps")
                    except Exception:
                        pass
                    try:
                        p2.metric('📤 Upload', f"{upload_speed:.2f} Mbps")
                    except Exception:
                        pass
                    try:
                        p3.metric('⚡ Ping', f"{ping_v:.2f} ms")
                    except Exception:
                        pass
                # Donut is rendered in the first column (`colg1`) above; no full-width duplicate here.
                # Detect which visualization backend is available and show a small badge
                vis_backend = 'none'
                try:
                    import plotly
                    vis_backend = 'plotly'
                except Exception:
                    try:
                        import altair
                        vis_backend = 'altair'
                    except Exception:
                        try:
                            import pandas
                            vis_backend = 'pandas'
                        except Exception:
                            vis_backend = 'streamlit'

                try:
                    st.caption(f'Visual ativo: {vis_backend}')
                except Exception:
                    pass
                

                # append to history and persist
                try:
                    entry = {'ts': datetime.now().isoformat(), 'download': download_speed, 'upload': upload_speed, 'ping': ping_v}
                    if 'speedtest_history' not in st.session_state:
                        st.session_state['speedtest_history'] = []
                    st.session_state['speedtest_history'].append(entry)
                    try:
                        import json as _json
                        with open('speedtest_history.json', 'w', encoding='utf-8') as _f:
                            _json.dump(st.session_state['speedtest_history'], _f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                except Exception:
                    pass

            except Exception as e:
                # Provide a clearer message for HTTP 403 Forbidden which is commonly
                # raised by urllib/requests when a test resource is blocked by the
                # server, CDN or ISP. Offer quick retry buttons and suggestions.
                try:
                    import urllib.error as _urllib_err
                except Exception:
                    _urllib_err = None

                serr = str(e or '')
                is_403 = False
                if _urllib_err and isinstance(e, _urllib_err.HTTPError):
                    try:
                        if getattr(e, 'code', None) == 403:
                            is_403 = True
                    except Exception:
                        pass
                if not is_403:
                    # Fallback textual detection
                    if '403' in serr or 'forbidden' in serr.lower() or 'http error 403' in serr.lower():
                        is_403 = True

                if is_403:
                    st.error('Erro: HTTP 403 (Forbidden) durante o teste de velocidade. Isso geralmente significa que um servidor/CDN/ISP bloqueou o download de recursos usados para medir a velocidade.')
                    st.markdown('Possíveis causas e ações:')
                    st.markdown("""- Seu provedor de internet ou firewall pode estar bloqueando os testes de velocidade.
""")
                    st.markdown('Tente as opções abaixo:')
                    st.markdown("""- Use o botão "🔁 Tentar novamente" para repetir o teste.
""")

                    colr, coll = st.columns(2)
                    with colr:
                        if st.button('🔁 Tentar novamente'):
                            # reset quick-mode toggle to previous selection
                            st.session_state['speedtest_quick_mode'] = False
                            st.experimental_rerun()
                    with coll:
                        if st.button('⚡ Tentar novamente (Modo Rápido)'):
                            st.session_state['speedtest_quick_mode'] = True
                            st.experimental_rerun()
                else:
                    st.error(f'Erro durante o teste de velocidade: {e}')

# ---------------------- Função Removedor de Fundo ---------------------- #
def background_remover():
    st.header('🖼️ Removedor de Fundo de Imagens')
    
    st.write('Faça upload de uma imagem para remover o fundo automaticamente. Você pode refinar o resultado manualmente usando a ferramenta de máscara (se disponível).')

    uploaded_file = st.file_uploader(
        "Escolha uma imagem",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formatos suportados: JPG, JPEG, PNG, BMP"
    )

    if uploaded_file is None:
        st.info("Envie uma imagem para começar.")
        return

    # Abrir imagem como RGBA
    try:
        input_image = Image.open(uploaded_file).convert('RGBA')
    except Exception as e:
        st.error(f"Erro ao abrir a imagem: {e}")
        return

    st.subheader("📷 Imagem Original")
    # Determine uploaded file size safely (UploadedFile may not expose size in all versions)
    try:
        size_bytes = getattr(uploaded_file, 'size', None)
        if size_bytes is None:
            # fallback to buffer length
            uploaded_file.seek(0)
            buf = uploaded_file.read()
            size_bytes = len(buf)
            uploaded_file.seek(0)
    except Exception:
        try:
            uploaded_file.seek(0)
            buf = uploaded_file.read()
            size_bytes = len(buf)
            uploaded_file.seek(0)
        except Exception:
            size_bytes = 0

    size_kb = round(size_bytes / 1024, 1) if size_bytes else 'N/A'
    st.caption(f"{input_image.width} x {input_image.height} px — {size_kb} KB")

    # Helper to render a gallery of images (original, initial output, mask, refined)
    def pil_to_bytes(img: Image.Image, fmt='PNG'):
        from io import BytesIO
        buf = BytesIO()
        try:
            if fmt.upper() == 'JPEG':
                rgb = img.convert('RGB')
                rgb.save(buf, format='JPEG', quality=90)
            else:
                img.save(buf, format=fmt)
        except Exception:
            img.convert('RGBA').save(buf, format=fmt)
        buf.seek(0)
        return buf.getvalue()

    def render_gallery(original, initial, mask_img, refined_img, refined_img2=None):
        # Build a responsive grid of thumbnails with captions and download buttons
        items = [
            ('Original', original),
            ('Resultado Inicial', initial),
            ('Máscara', mask_img),
        ]

        # If two refined variants are provided, show both with descriptive titles
        if refined_img is not None:
            items.append(('Refinado (com fundo)', refined_img))
        if refined_img2 is not None:
            items.append(('Refinado (sem fundo)', refined_img2))

        # Only include non-None items
        visible = [(t, i) for (t, i) in items if i is not None]
        if not visible:
            st.info('Nenhuma imagem para mostrar.')
            return

        cols_per_row = min(4, len(visible))
        cols = st.columns(cols_per_row)

        for idx, (title, img) in enumerate(visible):
            col = cols[idx % cols_per_row]
            with col:
                # Prepare thumbnail (preserve transparency when present)
                thumb = img.copy()
                thumb.thumbnail((420, 420))

                # Decide output format: prefer PNG when image has transparency
                try:
                    has_alpha = (thumb.mode in ('RGBA', 'LA')) or (thumb.mode == 'P' and 'transparency' in thumb.info)
                except Exception:
                    has_alpha = False

                fmt = 'PNG' if has_alpha else 'JPEG'
                # Prepare display bytes
                try:
                    disp_bytes = pil_to_bytes(thumb, fmt=fmt)
                except Exception:
                    disp_bytes = pil_to_bytes(thumb, fmt='PNG')

                # Robust equality check vs original: use numpy if available for pixel-wise comparison
                note = ''
                try:
                    if original is not None:
                        try:
                            import numpy as _np
                            a = _np.array(img.convert('RGBA'))
                            barr = _np.array(original.convert('RGBA'))
                            if a.shape == barr.shape and _np.array_equal(a, barr):
                                note = ' (igual ao Original)'
                        except Exception:
                            # Fallback to raw bytes compare
                            if img.size == original.size and img.convert('RGBA').tobytes() == original.convert('RGBA').tobytes():
                                note = ' (igual ao Original)'
                except Exception:
                    note = ''

                # Render inline image via data URL to avoid /media temp files
                try:
                    import base64 as _base64
                    b64 = _base64.b64encode(disp_bytes).decode('ascii')
                    mime = 'image/png' if fmt == 'PNG' else 'image/jpeg'
                    img_html = f'<img src="data:{mime};base64,{b64}" style="max-width:100%; height:auto;" alt="{title}" />'
                    st.markdown(img_html, unsafe_allow_html=True)
                    st.caption(f"{title}{note}")
                except Exception:
                    try:
                        st.image(disp_bytes, caption=f"{title}{note}", use_container_width=True)
                    except Exception:
                        st.write(f"[{title}] (não foi possível mostrar a miniatura)")

                # Download link with appropriate filename and mime
                try:
                    import base64 as _base64
                    full_bytes = pil_to_bytes(img, fmt=fmt)
                    b64_full = _base64.b64encode(full_bytes).decode('ascii')
                    filename = f"{title.replace(' ', '_')}.{ 'png' if fmt=='PNG' else 'jpg' }"
                    href = f'data:{"image/png" if fmt=="PNG" else "image/jpeg"};base64,{b64_full}'
                    # Use specific download labels requested by the user
                    label_map = {
                        'Original': '📥 Baixar Original',
                        'Resultado Inicial': '📥 Baixar Resultado Inicial',
                        'Refinado': '📥 Baixar Refinado',
                        'Refinado (com fundo)': '📥 Baixar Refinado (com fundo)',
                        'Refinado (sem fundo)': '📥 Baixar Refinado (sem fundo)'
                    }
                    dl_label = label_map.get(title, f'📥 Baixar {title}')
                    link_html = f'<a href="{href}" download="{filename}">{dl_label}</a>'
                    st.markdown(link_html, unsafe_allow_html=True)
                except Exception:
                    try:
                        import uuid as _uuid
                        mime = 'image/png' if fmt == 'PNG' else 'image/jpeg'
                        dl_key = f"gallery_dl_{idx}_{_uuid.uuid4().hex[:8]}"
                        st.download_button(label=dl_label, data=pil_to_bytes(img, fmt='PNG' if fmt=='PNG' else 'JPEG'), file_name=filename, mime=mime, key=dl_key)
                    except Exception:
                        st.write(f"[Baixar {title}] (não disponível)")

    # Robust rerun helper: use experimental_rerun when available, otherwise toggle a query param
    def safe_rerun():
        try:
            # Prefer the direct API
            rerun = getattr(st, 'experimental_rerun', None)
            if callable(rerun):
                return rerun()
        except Exception:
            pass
        try:
            # Fallback: update st.query_params in modern Streamlit
            try:
                params = dict(st.query_params)
                params['_reload'] = [str(uuid.uuid4())]
                st.query_params = params
                st.info('Rerun solicitado via query param; se nada acontecer, atualize manualmente a página (F5).')
            except Exception:
                # Final fallback: ask the user to refresh
                st.info('Por favor, recarregue a página manualmente (F5) para aplicar a alteração.')
        except Exception:
            st.info('Por favor, recarregue a página manualmente (F5) para aplicar a alteração.')

        pass

    # Ensure output_image exists (may be set later if rembg runs); use a copy of input as default
    output_image = input_image.copy()

    # Opções
    col_a, col_b = st.columns([1, 3])
    with col_a:
        do_auto = st.checkbox('🔮 Remover fundo automaticamente (rembg)', value=REMBG_AVAILABLE)
        if do_auto and not REMBG_AVAILABLE:
            st.warning('rembg não está instalado. A remoção automática não estará disponível.')
        quality = st.selectbox('Qualidade (afeta tempo e tamanho)', ['standard', 'high'], index=0)
        enable_manual = st.checkbox('✏️ Habilitar refinamento manual (máscara)', value=CANVAS_AVAILABLE)
        if enable_manual and not CANVAS_AVAILABLE:
            st.info('Instale streamlit-drawable-canvas para habilitar refinamento: pip install streamlit-drawable-canvas')

    # Processamento inicial: auto remove or fallback to original
    if do_auto and REMBG_AVAILABLE:
        with st.spinner('Removendo fundo automaticamente...'):
            try:
                # Optionally resize small copy for performance and then scale back
                max_dim = max(input_image.width, input_image.height)
                scale = 1.0
                temp_img = input_image
                if max_dim > 1600:
                    scale = 1600 / max_dim
                    temp_img = input_image.resize((int(input_image.width * scale), int(input_image.height * scale)))

                # Enhance slightly
                enhancer = ImageEnhance.Contrast(temp_img)
                enhanced = enhancer.enhance(1.2)

                # Call rembg and coerce result to a PIL Image
                from io import BytesIO
                raw = rembg_remove(enhanced)
                if isinstance(raw, (bytes, bytearray)):
                    output_image = Image.open(BytesIO(raw))
                elif isinstance(raw, Image.Image):
                    output_image = raw
                else:
                    try:
                        output_image = Image.open(str(raw))
                    except Exception:
                        output_image = input_image.copy()

                # Ensure RGBA
                try:
                    output_image = output_image.convert('RGBA')
                except Exception:
                    try:
                        import numpy as _np
                        output_image = Image.fromarray(_np.array(output_image)).convert('RGBA')
                    except Exception:
                        output_image = input_image.copy()

                # If we scaled down, scale back up
                if scale != 1.0:
                    output_image = output_image.resize((input_image.width, input_image.height))

                # Diagnostics (non-fatal): check whether anything changed or alpha present
                try:
                    import numpy as _np
                    orig_arr = _np.array(input_image.convert('RGBA'))
                    out_arr = _np.array(output_image.convert('RGBA'))
                    diff = (_np.abs(orig_arr.astype('int16') - out_arr.astype('int16'))).sum()
                    alpha_channel = out_arr[:, :, 3]
                    if diff == 0 or _np.all(alpha_channel == 255):
                        st.warning('Atenção: o resultado automático não parece ter removido o fundo (resultado igual ao original ou sem canal alfa). Verifique a instalação/versão do rembg ou tente outra imagem.')
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Erro na remoção automática: {e}")
                output_image = input_image.copy()
    else:
        output_image = input_image.copy()

    # Render gallery after processing to show the actual result
    # Pass two refined variants (same image) so the 'Refinado (com fundo)' slot is shown
    render_gallery(input_image, output_image, None, output_image, output_image)

    # The initial result is shown in the gallery above; further updates will rerender gallery.

    # Manual refine via canvas if available and enabled
    refined_image = output_image.copy()
    mask_image = None

    if enable_manual:
        # If the canvas package was not imported successfully earlier, inform the user
        if not CANVAS_AVAILABLE:
            st.warning('Ferramenta de refinamento manual não disponível. Instale streamlit-drawable-canvas para usar essa funcionalidade: pip install streamlit-drawable-canvas')
            st.info('Se já instalou o pacote, certifique-se de executar o Streamlit no mesmo ambiente virtual onde instalou as dependências.')
        else:
            st.markdown('Use o quadro abaixo para desenhar sobre a imagem. Use o menu para alternar entre "Apagar" (criar máscara para remover) ou "Restaurar" (restaurar partes da imagem original).')

            # Try calling the canvas and catch compatibility errors (some versions of streamlit / canvas
            # expect internals that changed and raise AttributeError like "image_to_url"). We must not let
            # the whole app crash because of that.
            try:
                # Prepare a data URL for the background image to avoid calling streamlit internals
                # (some canvas versions call streamlit.elements.image.image_to_url which may be missing).
                # Prepare a display-sized copy for the canvas background and for
                # debug rendering. Keep display_image in scope for use as a
                # background_image candidate as well. Use input_image (original) for canvas background.
                from io import BytesIO
                import base64
                display_image = input_image.copy()  # Use original image for canvas background
                try:
                    # Create a resized display image that fits the canvas dims
                    # Slightly smaller display caps to reduce data-URL size and JS stress
                    disp_w = min(input_image.width, 900)
                    disp_h = min(input_image.height, 600)
                    # Preserve aspect ratio
                    ratio = min(disp_w / input_image.width, disp_h / input_image.height, 1.0)
                    if ratio < 1.0:
                        disp_size = (max(1, int(input_image.width * ratio)), max(1, int(input_image.height * ratio)))
                        display_image = input_image.resize(disp_size, resample=Image.LANCZOS)
                    else:
                        display_image = input_image.copy()

                    buf_b = BytesIO()
                    # Use PNG to preserve alpha for the display copy
                    display_image.save(buf_b, format='PNG')
                    buf_b.seek(0)
                    b64 = base64.b64encode(buf_b.read()).decode('ascii')
                    data_url = f"data:image/png;base64,{b64}"
                    data_url_size_kb = round(len(b64) * 3 / 4 / 1024, 1)
                except Exception:
                    data_url = None
                    data_url_size_kb = None

                # Also create a persistent, disk-backed image file to avoid transient
                # media handler missing-file errors in some environments. We keep track
                # of these files to clean them up at the end of the request.
                bg_temp_files = []
                try:
                    import uuid as _uuid
                    import pathlib
                    cache_dir = pathlib.Path(os.path.join(os.getcwd(), 'canvas_bg_cache'))
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    bg_path = cache_dir / f"canvas_bg_{_uuid.uuid4().hex}.png"
                    # Save a PNG copy to disk (keeps file visible for MediaFileHandler)
                    # Use input_image (original) for canvas background
                    input_image.save(str(bg_path), format='PNG')
                    bg_temp_files.append(str(bg_path))
                except Exception:
                    bg_path = None

                # Canvas parameters (use background_image_url when available)
                # Match canvas dimensions exactly to the display image to avoid
                # backend/frontend scaling issues. Cap them to sensible maxima.
                max_canvas_w = 1000
                max_canvas_h = 800
                # Prefer the display-sized copy's dimensions (we prepared it above)
                try:
                    canvas_width = min(max_canvas_w, display_image.width)
                    canvas_height = min(max_canvas_h, display_image.height)
                except Exception:
                    canvas_width = min(max_canvas_w, output_image.width)
                    canvas_height = min(max_canvas_h, output_image.height)

                # Small control: allow user to force the canvas to retry rendering the
                # background using alternate strategies (disk-backed or in-memory PIL).
                # This helps when some Streamlit canvas backends ignore data-URL backgrounds.
                if 'canvas_force_count' not in st.session_state:
                    st.session_state['canvas_force_count'] = 0

                cols_force = st.columns([1, 8])
                with cols_force[0]:
                    if st.button("🔁 Forçar imagem", key="force_canvas_button"):
                        st.session_state['canvas_force_count'] = st.session_state.get('canvas_force_count', 0) + 1
                        # Trigger a rerun so the canvas re-attempts with a different strategy
                        st.experimental_rerun()

                canvas_kwargs = dict(
                    fill_color="rgba(255, 0, 0, 0.5)",
                    stroke_width=20,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="freedraw",
                    key="bg_remove_canvas",
                )

                # Try calling st_canvas with multiple strategies and progressively remove
                # problematic kwargs if TypeError reports unexpected keyword arguments.
                canvas_result = None
                canvas_exc = None

                # Decide whether to prefer data-URL backgrounds or image objects.
                # If the data-URL is very large, try image-based backgrounds first
                # because some canvas backends choke on large data URLs.
                data_url_kb = None
                if data_url:
                    try:
                        b64 = data_url.split(',', 1)[1]
                        approx_bytes = (len(b64) * 3) / 4
                        data_url_kb = approx_bytes / 1024.0
                    except Exception:
                        data_url_kb = None

                # Default: prefer data-URL unless it's large (> 700 KB)
                # or the user pressed the force button (which prefers image objects).
                if st.session_state.get('canvas_force_count', 0) > 0:
                    force_data_url = False
                else:
                    if data_url_kb is None:
                        force_data_url = True
                    else:
                        force_data_url = False if data_url_kb > 700 else True

                # Candidate order: data URL (preferred), disk-backed PIL image, display_image, in-memory PIL image, no background
                attempts = []
                if force_data_url and data_url:
                    # Prefer trying the data URL first; do NOT try an empty/no-background
                    # attempt immediately afterwards because some canvas builds will
                    # accept a blank canvas (method 'none') and stop the attempts
                    # before image-based strategies are tried.
                    attempts = [{'background_image_url': data_url}]
                else:
                    if data_url:
                        attempts.append({'background_image_url': data_url})
                # Prefer a disk-backed image to avoid transient media file errors
                if bg_path is not None:
                    try:
                        # Pass a PIL.Image opened from the disk-backed file
                        from PIL import Image as _PILImage
                        disk_img = _PILImage.open(str(bg_path)).convert('RGB')
                        attempts.append({'background_image': disk_img})
                    except Exception:
                        pass
                # include the display-sized PIL image as a candidate (works better for many canvas versions)
                try:
                    attempts.append({'background_image': display_image.convert('RGB')})
                except Exception:
                    pass
                # If the user requested, try passing a numpy ndarray background (some canvas versions accept this)
                try:
                    if st.session_state.get('canvas_try_ndarray', False):
                        import numpy as _np
                        disp_arr = _np.array(display_image.convert('RGB'))
                        # Ensure dtype uint8
                        if disp_arr.dtype != _np.uint8:
                            disp_arr = disp_arr.astype(_np.uint8)
                        attempts.insert(0, {'background_image': disp_arr})
                except Exception:
                    pass
                # In-memory PIL fallback
                attempts.append({'background_image': input_image})  # Use original image

                # If the user explicitly asked to force the image, prefer the in-memory
                # PIL background first (this increases likelihood the canvas will render it).
                if st.session_state.get('canvas_force_count', 0) > 0:
                    try:
                        # Prepend a prioritised in-memory background attempt so it's tried first
                        attempts.insert(0, {'background_image': input_image})
                        # Keep a subtle info message so users know their request was applied
                        st.info("🔁 Tentativa forçada: priorizando background como imagem em memória...")
                    except Exception:
                        pass
                # finally, no background image (try this only after all image-based attempts)
                attempts.append({})

                canvas_used_method = None
                for attempt_kwargs in attempts:
                    trial_kwargs = dict(canvas_kwargs)
                    trial_kwargs.update(attempt_kwargs)
                    try:
                        canvas_result = st_canvas(**trial_kwargs)
                        # Record which method succeeded
                        if 'background_image_url' in trial_kwargs:
                            canvas_used_method = 'background_image_url'
                        elif 'background_image' in trial_kwargs:
                            canvas_used_method = 'background_image'
                        else:
                            canvas_used_method = 'none'
                        # If we get here, success
                        break
                    except TypeError as te:
                        msg = str(te).lower()
                        # If message indicates unexpected keyword, try next attempt
                        if 'unexpected' in msg and 'keyword' in msg:
                            canvas_exc = te
                            continue
                        else:
                            # Other TypeError: capture and break
                            canvas_exc = te
                            break
                    except Exception as e:
                        # Specific known incompatibility earlier raised AttributeError
                        canvas_exc = e
                        # If AttributeError mentions image_to_url, try next strategy
                        if isinstance(e, AttributeError) and 'image_to_url' in str(e):
                            continue
                        # Otherwise try next strategy
                        continue

                # If nothing worked or canvas accepted no background, optionally attempt
                # passing a numpy ndarray as a last resort but only if the user explicitly
                # requested this path (we don't want to trigger fragile ndarray handling
                # automatically because some canvas builds raise confusing ValueError
                # when they try to test ndarray objects in boolean contexts).
                if (canvas_result is None or canvas_used_method == 'none') and st.session_state.get('canvas_try_ndarray', False):
                    try:
                        import numpy as _np
                        arr = _np.array(display_image.convert('RGB'))
                        if arr.dtype != _np.uint8:
                            arr = arr.astype(_np.uint8)
                        trial_kwargs = dict(canvas_kwargs)
                        trial_kwargs.update({'background_image': arr})
                        canvas_result = st_canvas(**trial_kwargs)
                        if canvas_result is not None:
                            canvas_used_method = 'background_image_ndarray'
                    except Exception as e:
                        canvas_exc = e

                # Initialize fallback variables
                fallback_mask_data = None
                fallback_active = False

                # Inform the user which background method succeeded (helps debugging)
                try:
                    if canvas_result is not None:
                        st.success(f"Canvas inicializado com método: {canvas_used_method}")
                    if data_url_kb is not None:
                        st.caption(f"Data-URL size (KB): {data_url_kb:.1f}")
                except Exception:
                    pass

                # Initialize fallback variables
                fallback_mask_data = None
                fallback_active = False
                
                # If canvas failed completely OR canvas succeeded but has no background, try HTML5 fallback
                if canvas_result is None or canvas_used_method == 'none':
                    st.info("🎨 Canvas principal não conseguiu mostrar a imagem. Usando canvas alternativo...")
                    
                    # Try the HTML5 fallback if we have a data_url or display_image to show
                    try:
                        if data_url:
                            # create a unique key to avoid duplicate component collisions
                            import uuid as _uuid
                            key = f'fallback_canvas_{_uuid.uuid4().hex[:8]}'
                            
                            # Try components.html first (remove key parameter as it's not supported)
                            try:
                                fallback_mask_data = html5_canvas_component(data_url, canvas_width, canvas_height)
                                if fallback_mask_data is not None:
                                    fallback_active = True
                                    # Canvas carregado sem mensagens de debug
                                else:
                                    st.warning("⚠️ Canvas alternativo não retornou dados")
                            except Exception as e:
                                st.error(f"❌ Erro no canvas alternativo: {e}")
                                
                            # If HTML5 component failed, show a basic image preview at least
                            if not fallback_active:
                                # Removido: prévia da imagem solta conforme solicitado pelo usuário
                                st.info("💡 Canvas interativo não disponível, mas você pode baixar o resultado através da galeria abaixo.")
                                pass
                        else:
                            st.error("❌ Nenhuma data-URL disponível para o canvas alternativo")
                    except Exception as e:
                        st.error(f"❌ Erro geral no canvas alternativo: {e}")
                        fallback_mask_data = None
                        fallback_active = False

                    if fallback_mask_data and isinstance(fallback_mask_data, str) and fallback_mask_data.startswith('data:image'):
                        # Decode returned data URL into PIL image and process it like canvas mask
                        try:
                            from io import BytesIO
                            import base64 as _base64
                            header, b64 = fallback_mask_data.split(',', 1)
                            mask_bytes = _base64.b64decode(b64)
                            mask_image = Image.open(BytesIO(mask_bytes)).convert('L')

                            # Resize mask to original image size
                            orig_w, orig_h = input_image.width, input_image.height
                            mask_resized = mask_image.resize((orig_w, orig_h), resample=Image.NEAREST)

                            try:
                                import numpy as np
                                mask_bool = np.array(mask_resized) > 127
                            except Exception:
                                # Fallback: treat any non-zero pixel as True
                                mask_bool = np.array(mask_resized) != 0

                            if mask_bool is not None and mask_bool.any():
                                # Ask the user whether the drawing is erase or restore
                                action = st.radio('Ação da máscara:', ('Apagar (tornar transparente)', 'Restaurar (trazer do original)'))

                                # Convert images to arrays for manipulation (ensure RGBA)
                                import numpy as np
                                orig_arr = np.array(input_image.convert('RGBA'))
                                out_arr = np.array(refined_image.convert('RGBA'))

                                if action.startswith('Apagar'):
                                    out_arr[mask_bool, 3] = 0
                                else:
                                    out_arr[mask_bool] = orig_arr[mask_bool]

                                refined_image = Image.fromarray(out_arr)

                                # Create two variants:
                                # 1) refined_with_bg: the refined image composited over the original background
                                try:
                                    base_rgba = input_image.convert('RGBA')
                                    refined_with_bg = Image.alpha_composite(base_rgba, refined_image.convert('RGBA'))
                                except Exception:
                                    refined_with_bg = refined_image.copy()

                                # 2) refined_no_bg: ensure alpha channel reflects mask removal (transparent background)
                                try:
                                    refined_no_bg = refined_image.convert('RGBA')
                                except Exception:
                                    refined_no_bg = refined_image.copy()

                                # Render gallery with both refined variants; remove extra overlay image
                                render_gallery(input_image, output_image, mask_resized, refined_with_bg, refined_no_bg)
                                # Skip the normal error path - fallback worked
                                # Don't set canvas_result to True as it causes issues later
                                # Just continue without setting canvas_result since fallback handled it
                                pass
                            else:
                                # No mask drawn; fall through to showing the error below
                                pass
                        except Exception as e:
                            canvas_exc = e

                    # Only show error messages if both main canvas AND fallback failed
                    if not fallback_active:
                        st.error('Falha ao inicializar a ferramenta de desenho: st_canvas não aceitou as opções testadas.')
                        if canvas_exc is not None:
                            st.write(f'Detalhe: {type(canvas_exc).__name__}: {str(canvas_exc)}')
                        # Report which candidate paths we tried to help debugging
                        tried = [list(a.keys()) for a in attempts]
                        st.write(f'Métodos testados: {tried}')
                        if bg_path:
                            st.write(f'Arquivo de background salvo em: {bg_path} (permanece até limpar)')
                        # Show any files present in the canvas_bg_cache directory to help
                        # diagnose MediaFileHandler missing-file errors observed in some setups.
                        try:
                            import pathlib
                            cache_dir = pathlib.Path(os.path.join(os.getcwd(), 'canvas_bg_cache'))
                            if cache_dir.exists():
                                files = [str(p.name) for p in cache_dir.iterdir()]
                                st.write(f'Arquivos no canvas_bg_cache: {files}')
                            else:
                                st.write('canvas_bg_cache não existe no diretório do app.')
                        except Exception as _:
                            pass
                    else:
                        # Hide the complex error messages since fallback worked
                        pass  # Canvas alternativo funcionando silenciosamente

                # If we got a canvas but it didn't accept any background (method 'none'),
                # offer the user a retry that passes the display image as a numpy ndarray
                # which some versions of the canvas accept. But skip this if we're already using HTML5 fallback.
                if canvas_result is not None and canvas_used_method == 'none' and not fallback_active:
                    st.warning('O Canvas foi inicializado sem background. Se não está vendo a imagem, tente a alternativa: passar o background como array (ndarray).')
                    if 'canvas_try_ndarray' not in st.session_state:
                        st.session_state.canvas_try_ndarray = False
                    if st.button('🔁 Tentar alternativa: usar ndarray como background'):
                        st.session_state.canvas_try_ndarray = True
                        safe_rerun()

                    # Offer a reload to attempt canvas initialization again
                    if st.button('🔁 Recarregar quadro'):
                        safe_rerun()
                    st.markdown('''
                    Possíveis soluções:
                    - Atualize/reinstale: `pip install --upgrade streamlit-drawable-canvas`
                    - Alinhe a versão do Streamlit com a do canvas (ex.: `pip install "streamlit==1.26.0"`)
                    - Reinicie o Streamlit a partir do mesmo ambiente virtual onde instalou as dependências
                    ''')
                    st.info('Enquanto isso, você pode baixar o resultado inicial mostrado acima ou reabrir esta página após corrigir as dependências.')

            except AttributeError as ae:
                # Known incompatibility in some versions: streamlit.elements.image.image_to_url missing
                st.error('Erro ao inicializar a ferramenta de desenho (incompatibilidade entre streamlit e streamlit-drawable-canvas).')
                st.write(f'Detalhe: {ae}')
                st.markdown('''
                Possíveis soluções:
                - Instale/atualize o pacote: `pip install --upgrade streamlit-drawable-canvas`
                - Ou ajuste a versão do Streamlit para a compatível com a sua versão do canvas (ex.: `pip install "streamlit==1.26.0"`).
                - Reinicie o Streamlit a partir do mesmo ambiente virtual onde fez a instalação.
                ''')
                st.info('Enquanto isso, você pode baixar o resultado inicial mostrado acima ou reabrir esta página após corrigir as dependências.')
                canvas_result = None

            except Exception as e:
                st.error('Falha ao inicializar a ferramenta de desenho:')
                st.write(str(e))
                st.info('Verifique a instalação de streamlit-drawable-canvas e se está usando o mesmo ambiente virtual.')
                canvas_result = None

            # If canvas_result is available, process the drawing as before
            if canvas_result is not None:
                # Report which method actually succeeded (helps debugging) - but skip if using HTML5 fallback
                if not fallback_active:
                    try:
                        st.write(f'Canvas inicializado com método: {canvas_used_method}')
                        if data_url_size_kb is not None:
                            st.write(f'Data-URL size (KB): {data_url_size_kb}')
                    except Exception:
                        pass

            # If canvas_result is available and is a proper canvas result object, process the drawing
            if canvas_result is not None and hasattr(canvas_result, 'image_data') and getattr(canvas_result, 'image_data', None) is not None:
                import numpy as np
                try:
                    canvas_arr = canvas_result.image_data  # HxWx4 RGBA
                    # Create mask from alpha channel or non-white pixels
                    # We treat non-transparent drawn pixels as mask
                    alpha = canvas_arr[:, :, 3]
                    mask = alpha > 10  # boolean mask where user drew
                    if mask.any():
                        # Create a mask image and resize it back to the original image size
                        # because the canvas may be rendered at different dimensions.
                        try:
                            mask_image = Image.fromarray((mask * 255).astype('uint8'))

                            orig_w, orig_h = input_image.width, input_image.height
                        except Exception:
                            mask_image = Image.fromarray((mask * 255).astype('uint8'))
                            orig_w, orig_h = input_image.width, input_image.height

                        # Resize mask to original image size using nearest neighbor to preserve binary mask
                        try:
                            mask_resized = mask_image.resize((orig_w, orig_h), resample=Image.NEAREST)
                            mask_bool = np.array(mask_resized) > 127
                        except Exception:
                            try:
                                mask_bool = mask
                            except Exception:
                                st.error('Erro ao processar a máscara desenhada. Tente novamente com uma imagem menor.')
                                mask_bool = None

                        if mask_bool is not None and mask_bool.any():
                            # Ask the user whether the drawing is erase or restore
                            action = st.radio('Ação da máscara:', ('Apagar (tornar transparente)', 'Restaurar (trazer do original)'))

                            # Convert images to arrays for manipulation (ensure RGBA)
                            orig_arr = np.array(input_image.convert('RGBA'))
                            out_arr = np.array(refined_image.convert('RGBA'))

                            if action.startswith('Apagar'):
                                # Set alpha to 0 where mask==True
                                out_arr[mask_bool, 3] = 0
                            else:
                                # Copy original RGBA pixels where mask==True
                                out_arr[mask_bool] = orig_arr[mask_bool]

                            refined_image = Image.fromarray(out_arr)

                            # Create two variants:
                            try:
                                base_rgba = input_image.convert('RGBA')
                                refined_with_bg = Image.alpha_composite(base_rgba, refined_image.convert('RGBA'))
                            except Exception:
                                refined_with_bg = refined_image.copy()

                            try:
                                refined_no_bg = refined_image.convert('RGBA')
                            except Exception:
                                refined_no_bg = refined_image.copy()

                            # Render gallery with both refined variants and do not render the overlay preview
                            render_gallery(input_image, output_image, mask_resized, refined_with_bg, refined_no_bg)
                except Exception as e:
                    st.error(f"Erro ao processar resultado do canvas: {e}")
                    # Continue with the app even if canvas processing fails

                # We will remove any background files we created after rendering the
                # download buttons below to ensure the MediaFileHandler can still read them
                # during this request. The cleanup happens at the end of the function.
                pass

    # If canvas not available but user wants manual refinement, inform them
    if enable_manual and not CANVAS_AVAILABLE:
        st.warning('Ferramenta de refinamento manual não disponível. Instale streamlit-drawable-canvas para usar essa funcionalidade: pip install streamlit-drawable-canvas')
        st.info('Enquanto a ferramenta não estiver disponível você ainda pode baixar o resultado inicial via o botão abaixo.')

    # Provide download button (PNG) that sends the image to the browser
    from io import BytesIO
    buf = BytesIO()
    try:
        refined_image.save(buf, format='PNG')
    except Exception:
        # fallback: convert to RGBA then save
        refined_image.convert('RGBA').save(buf, format='PNG')
    buf.seek(0)

    try:
        import base64 as _base64
        png_b = buf.getvalue()
        png_b64 = _base64.b64encode(png_b).decode('ascii')
        png_href = f'data:image/png;base64,{png_b64}'
        st.markdown(f'<a href="{png_href}" download="imagem_sem_fundo.png">📥 Baixar PNG (transparente)</a>', unsafe_allow_html=True)
    except Exception:
        try:
            st.download_button(label='📥 Baixar PNG (transparente)', data=buf.getvalue(), file_name='imagem_sem_fundo.png', mime='image/png')
        except Exception:
            st.write('📥 Baixar PNG não disponível')

    # Option: also offer JPG (no transparency)
    try:
        buf_jpg = BytesIO()
        rgb = refined_image.convert('RGB')
        rgb.save(buf_jpg, format='JPEG', quality=90)
        buf_jpg.seek(0)
        jpg_b = buf_jpg.getvalue()
        jpg_b64 = _base64.b64encode(jpg_b).decode('ascii')
        jpg_href = f'data:image/jpeg;base64,{jpg_b64}'
        st.markdown(f'<a href="{jpg_href}" download="imagem_sem_fundo.jpg">📥 Baixar JPG (fundo branco)</a>', unsafe_allow_html=True)
    except Exception:
        try:
            st.download_button(label='📥 Baixar JPG (fundo branco)', data=buf_jpg.getvalue(), file_name='imagem_sem_fundo.jpg', mime='image/jpeg')
        except Exception:
            st.write('📥 Baixar JPG não disponível')

    # Cleanup any background files created for the canvas (if any)
    try:
        if 'bg_temp_files' in locals() and bg_temp_files:
            for f in bg_temp_files:
                safe_remove_file(f)
            # Try to remove the cache directory if empty
            try:
                import pathlib
                cache_dir = pathlib.Path(os.path.join(os.getcwd(), 'canvas_bg_cache'))
                if cache_dir.exists() and not any(cache_dir.iterdir()):
                    cache_dir.rmdir()
            except Exception:
                pass
    except Exception:
        pass

# ---------------------- Página Inicial ---------------------- #
def home_page():
    st.title("🚀 Central de Ferramentas Multimídia")
    
    st.markdown("""
    ### 👋 Bem-vindo à nossa plataforma completa!
    
    Esta aplicação oferece diversas ferramentas úteis para você:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎬 **Baixar Vídeos do YouTube**
        - Download de vídeos em alta qualidade (MP4)
        - Extração de áudio (MP3)
        - Suporte a múltiplas URLs
        - Lista de downloads organizada
        """)
        
        st.markdown("""
        ### 🎵 **Transcrição de Áudio**
        - Transcrição automática de vídeos/áudios
        - Geração de PDF com transcrição
        - Suporte a múltiplos formatos
        - Reconhecimento em português brasileiro
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 **Teste de Velocidade**
        - Velocidade de download
        - Velocidade de upload  
        - Medição de ping/latência
        - Informações do servidor
        """)
        
        st.markdown("""
        ### 🖼️ **Removedor de Fundo**
        - Remoção automática de fundo de imagens
        - Suporte a diversos formatos
        - Download da imagem processada
        - Tecnologia de IA avançada
        """)
    
    st.markdown("""
    ---
    ### 🎯 **Como usar:**
    1. **Escolha uma ferramenta** no menu lateral
    2. **Siga as instruções** de cada seção
    3. **Aproveite os resultados!**
    
    ### 💡 **Dicas:**
    - Para vídeos do YouTube, separe múltiplas URLs com vírgula
    - Para melhor qualidade na remoção de fundo, use imagens nítidas
    - O teste de velocidade pode levar alguns minutos
    - As transcrições funcionam melhor com áudio claro
    """)

def tts_page():
    """Página de Text-to-Speech: cole texto, escolha provedor/voz e gere áudio reproduzível e para download."""
    st.header('🔊 Text-to-Speech (TTS)')

    st.write('Cole o texto no campo abaixo e escolha a voz/serviço. O app tentará usar um TTS mais realista disponível (edge-tts), senão recairá para gTTS ou pyttsx3 offline.')

    # Use module-level availability flags set at import time
    global EDGE_TTS_AVAILABLE, GTTTS_AVAILABLE, PYTTSX3_AVAILABLE

    providers = []
    if EDGE_TTS_AVAILABLE:
        providers.append('edge-tts (mais natural)')
    if GTTTS_AVAILABLE:
        providers.append('gTTS (Google)')
    if PYTTSX3_AVAILABLE:
        providers.append('pyttsx3 (offline)')
    if not providers:
        providers = ['Nenhum provedor disponível']

    provider = st.selectbox('Provedor TTS', providers)

    # Voice presets (user asked for 'Liam')
    voice_preset = st.selectbox('Preset de voz (rápido)', ['Nenhum', 'Liam (LLM)'])

    # Recommended voices dropdown (common natural voices - edge-tts identifiers)
    recommended_voices = [
        # Português
        'pt-BR-FranciscaNeural',
        'pt-BR-AntonioNeural',
        'pt-BR-RaissaNeural',
        'pt-BR-HeloisaRUS',
        'pt-BR-Daniel',
        'pt-BR-FranciscaRUS',
        'pt-BR-HumbertoRUS',
        'pt-PT-FernandaNeural',
        'pt-PT-RaquelNeural',
        # Inglês
        'en-US-LiamNeural',
        'en-GB-LiamNeural',
        'en-US-JennyNeural',
        'en-US-GuyNeural',
        'en-US-AriaNeural',
        'en-US-ZiraNeural',
        'en-GB-SoniaNeural',
        'en-GB-RyanNeural',
        'en-AU-NathanNeural',
        'en-CA-LiamNeural',
        'en-IN-NeerjaNeural',
        'en-IN-PrabhatNeural',
        # Espanhol
        'es-ES-ElviraNeural',
        'es-MX-DaliaNeural',
        'es-AR-ElenaNeural',
        'es-CO-SalomeNeural',
        'es-US-PalomaNeural',
        'es-US-AlonsoNeural',
        # Francês
        'fr-FR-DeniseNeural',
        'fr-CA-SylvieNeural',
        'fr-BE-CharlineNeural',
        'fr-CH-ArianeNeural',
        # Alemão
        'de-DE-KatjaNeural',
        'de-DE-ConradNeural',
        'de-AT-IngridNeural',
        'de-CH-LeniNeural',
        # Italiano
        'it-IT-ElsaNeural',
        'it-IT-DiegoNeural',
        # Japonês
        'ja-JP-NanamiNeural',
        'ja-JP-KeitaNeural',
        # Chinês
        'zh-CN-XiaoxiaoNeural',
        'zh-CN-YunyangNeural',
        'zh-CN-XiaohanNeural',
        'zh-CN-XiaomoNeural',
        'zh-CN-XiaoxuanNeural',
        'zh-CN-XiaoyouNeural',
        'zh-TW-HsiaoChenNeural',
        'zh-TW-YunJheNeural',
        'zh-HK-HiuMaanNeural',
        'zh-HK-WanLungNeural',
        # Coreano
        'ko-KR-SunHiNeural',
        'ko-KR-InJoonNeural',
        # Árabe
        'ar-SA-ZariyahNeural',
        'ar-SA-HamedNeural',
        'ar-EG-SalmaNeural',
        'ar-EG-ShakirNeural',
        # Russo
        'ru-RU-SvetlanaNeural',
        'ru-RU-DmitryNeural',
        # Holandês
        'nl-NL-ColetteNeural',
        'nl-NL-MaartenNeural',
        # Sueco
        'sv-SE-SofieNeural',
        'sv-SE-MattiasNeural',
        # Norueguês
        'nb-NO-PernilleNeural',
        'nb-NO-FinnNeural',
        # Dinamarquês
        'da-DK-ChristelNeural',
        'da-DK-JeppeNeural',
        # Finlandês
        'fi-FI-SelmaNeural',
        'fi-FI-HarriNeural',
        # Polonês
        'pl-PL-AgnieszkaNeural',
        'pl-PL-MarekNeural',
        # Turco
        'tr-TR-EmelNeural',
        'tr-TR-AhmetNeural',
        # Tcheco
        'cs-CZ-VlastaNeural',
        'cs-CZ-AntoninNeural',
        # Húngaro
        'hu-HU-NoemiNeural',
        'hu-HU-TamasNeural',
        # Grego
        'el-GR-AthinaNeural',
        'el-GR-NestorasNeural',
        # Hebraico
        'he-IL-HilaNeural',
        'he-IL-AvriNeural',
        # Hindi
        'hi-IN-SwaraNeural',
        'hi-IN-MadhurNeural',
        # Tâmil
        'ta-IN-PallaviNeural',
        'ta-IN-ValluvarNeural',
        # Telugu
        'te-IN-ShrutiNeural',
        'te-IN-MohanNeural',
        # Canadense
        'en-CA-ClaraNeural',
        'en-CA-LiamNeural'
    ]
    selected_recommended = st.selectbox('Voz recomendada (testar rapidamente)', recommended_voices)
    
    # Voice language filter
    voice_languages = {
        '🇧🇷 Português': [v for v in recommended_voices if v.startswith('pt-')],
        '🇺🇸 Inglês': [v for v in recommended_voices if v.startswith('en-')],
        '🇪🇸 Espanhol': [v for v in recommended_voices if v.startswith('es-')],
        '🇫🇷 Francês': [v for v in recommended_voices if v.startswith('fr-')],
        '🇩🇪 Alemão': [v for v in recommended_voices if v.startswith('de-')],
        '🇮🇹 Italiano': [v for v in recommended_voices if v.startswith('it-')],
        '🇯🇵 Japonês': [v for v in recommended_voices if v.startswith('ja-')],
        '🇨🇳 Chinês': [v for v in recommended_voices if v.startswith('zh-')],
        '🇰🇷 Coreano': [v for v in recommended_voices if v.startswith('ko-')],
        '🇸🇦 Árabe': [v for v in recommended_voices if v.startswith('ar-')],
        '🇷🇺 Russo': [v for v in recommended_voices if v.startswith('ru-')],
        '🇳🇱 Holandês': [v for v in recommended_voices if v.startswith('nl-')],
        '🇸🇪 Sueco': [v for v in recommended_voices if v.startswith('sv-')],
        '🇳🇴 Norueguês': [v for v in recommended_voices if v.startswith('nb-')],
        '🇩🇰 Dinamarquês': [v for v in recommended_voices if v.startswith('da-')],
        '🇫🇮 Finlandês': [v for v in recommended_voices if v.startswith('fi-')],
        '🇵🇱 Polonês': [v for v in recommended_voices if v.startswith('pl-')],
        '🇹🇷 Turco': [v for v in recommended_voices if v.startswith('tr-')],
        '🇨🇿 Tcheco': [v for v in recommended_voices if v.startswith('cs-')],
        '🇭🇺 Húngaro': [v for v in recommended_voices if v.startswith('hu-')],
        '🇬🇷 Grego': [v for v in recommended_voices if v.startswith('el-')],
        '🇮🇱 Hebraico': [v for v in recommended_voices if v.startswith('he-')],
        '🇮🇳 Hindi': [v for v in recommended_voices if v.startswith('hi-')],
        '🇮🇳 Tâmil': [v for v in recommended_voices if v.startswith('ta-')],
        '🇮🇳 Telugu': [v for v in recommended_voices if v.startswith('te-')],
        '🌍 Todas as vozes': recommended_voices
    }
    
    selected_language = st.selectbox('🌍 Filtrar por idioma:', list(voice_languages.keys()))
    filtered_voices = voice_languages[selected_language]
    
    # Show voice statistics
    total_voices = len(recommended_voices)
    total_languages = len([k for k in voice_languages.keys() if k != '🌍 Todas as vozes'])
    
    st.info(f'📊 **{total_voices} vozes gratuitas** disponíveis em **{total_languages} idiomas** | Filtrando: {len(filtered_voices)} vozes de {selected_language}')

    # Map presets to voice identifiers for edge-tts; fallbacks used if not available
    def preset_to_voice(preset):
        if preset == 'Liam (LLM)':
            # We will attempt several candidate voice identifiers for 'Liam'
            # Edge TTS typically uses names like en-US-LiamNeural or en-GB-LiamNeural
            return 'en-US-LiamNeural'
        return ''

    text = st.text_area('Texto para converter em fala', height=200)
    filename = st.text_input('Nome do arquivo (sem extensão)', value='tts_output')

    col1, col2 = st.columns([1,1])
    with col1:
        # Determine default voice value
        default_voice = ''
        if selected_recommended:
            default_voice = selected_recommended
        elif voice_preset and voice_preset != 'Nenhum':
            default_voice = preset_to_voice(voice_preset)
        else:
            default_voice = ('en-US-LiamNeural' if EDGE_TTS_AVAILABLE else ('en' if not GTTTS_AVAILABLE else 'en'))

        voice = st.text_input('Voz / Idioma (opcional)', value=default_voice)
    with col2:
        fmt = st.selectbox('Formato', ['mp3', 'wav'])

    if st.button('▶️ Gerar Áudio'):
        if not text.strip():
            st.error('Por favor cole algum texto.')
        else:
            with st.spinner('Gerando áudio...'):
                audio_bytes = None
                out_name = f"{filename}.{fmt}"

                # edge-tts
                if 'edge-tts' in provider and EDGE_TTS_AVAILABLE:
                    try:
                        import asyncio
                        import edge_tts

                        async def synthesize_candidates(text, voices, out_fmt):
                            # Try each voice id until one succeeds; return data and voice used
                            for v in voices:
                                try:
                                    communicate = edge_tts.Communicate(text, v)
                                    tmp = f"tmp_tts_{uuid.uuid4().hex[:8]}.{out_fmt}"
                                    await communicate.save(tmp)
                                    # ensure file exists and is non-empty
                                    if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                                        with open(tmp, 'rb') as f:
                                            data = f.read()
                                        safe_remove_file(tmp)
                                        return data, v
                                    else:
                                        safe_remove_file(tmp)
                                        # try next candidate
                                        continue
                                except Exception:
                                    # try next voice
                                    continue
                            raise RuntimeError('Nenhuma voice candidate funcionou')

                        # Build candidate list: prioritize explicit recommended selection
                        if selected_recommended:
                            candidate_voices = [selected_recommended]
                        elif voice_preset == 'Liam (LLM)':
                            # Try many Liam-like candidates and known-good voices
                            candidate_voices = [
                                'en-GB-LiamNeural',
                                'en-US-LiamNeural',
                                'en-CA-LiamNeural',
                                'en-US-GuyNeural',
                                'en-US-JennyNeural'
                            ]
                        else:
                            # if user typed a voice use it first otherwise try common known-good voices
                            candidate_voices = [voice] if voice else [
                                'pt-BR-FranciscaNeural',
                                'pt-BR-AntonioNeural',
                                'en-US-JennyNeural',
                                'en-US-GuyNeural'
                            ]

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            provider_used = 'edge-tts'
                            result = loop.run_until_complete(synthesize_candidates(text, candidate_voices, fmt))
                            if result is not None:
                                audio_bytes, voice_used = result
                            else:
                                audio_bytes, voice_used = None, None
                        finally:
                            loop.close()
                    except Exception as e:
                        st.warning(f'Falha edge-tts: {e}')
                        audio_bytes = None

                # gTTS fallback
                if audio_bytes is None and 'gTTS' in provider and GTTTS_AVAILABLE:
                    try:
                        from gtts import gTTS
                        t = gTTS(text=text, lang=voice.split('-')[0] if '-' in voice else voice)
                        from io import BytesIO
                        buf = BytesIO()
                        t.write_to_fp(buf)
                        buf.seek(0)
                        audio_bytes = buf.read()
                        provider_used = 'gTTS'
                        voice_used = f'gTTS({voice})'
                    except Exception as e:
                        st.warning(f'Falha gTTS: {e}')
                        audio_bytes = None

                # pyttsx3 offline fallback
                if audio_bytes is None and PYTTSX3_AVAILABLE:
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        # Try to pick a voice matching 'liam' if available
                        try:
                            voices = engine.getProperty('voices')
                            chosen = None
                            for v in voices:
                                name = (v.name or '') + ' ' + (getattr(v, 'id', '') or '')
                                if 'liam' in name.lower():
                                    chosen = v.id
                                    break
                            if chosen:
                                engine.setProperty('voice', chosen)
                        except Exception:
                            pass

                        tmp = f"tmp_tts_{uuid.uuid4().hex[:8]}.mp3"
                        engine.save_to_file(text, tmp)
                        engine.runAndWait()
                        with open(tmp, 'rb') as f:
                            audio_bytes = f.read()
                        provider_used = 'pyttsx3'
                        voice_used = chosen if 'chosen' in locals() and chosen else 'pyttsx3_default'
                        safe_remove_file(tmp)
                    except Exception as e:
                        st.warning(f'Falha pyttsx3: {e}')
                        audio_bytes = None

                if audio_bytes and len(audio_bytes) > 0:
                    st.success('Áudio gerado com sucesso!')
                    # Show which provider/voice produced the audio if available
                    prov = locals().get('provider_used', provider)
                    vused = locals().get('voice_used', voice if voice else selected_recommended)
                    st.info(f'Provedor usado: {prov} | Voz: {vused}')
                    # Play inline (streamlit supports audio)
                    st.audio(audio_bytes, format=f"audio/{fmt}")

                    # Download button
                    st.download_button('📥 Baixar Áudio', data=audio_bytes, file_name=out_name, mime=f"audio/{fmt}")
                else:
                    st.error('Não foi possível gerar áudio com os provedores disponíveis. Instale edge-tts ou gTTS e verifique se o provedor está acessível no ambiente.')

    # Show instructions for installing providers
    with st.expander('Instalação / Dicas (opcional)'):
        st.markdown('''
        ### 🎯 **Sobre as Vozes Disponíveis:**
        - **80+ vozes gratuitas** em **35+ idiomas** da Microsoft Edge TTS
        - Qualidade neural de alta fidelidade (sem custos!)
        - Suporte a idiomas: Português, Inglês, Espanhol, Francês, Alemão, Italiano, Japonês, Chinês, Coreano, Árabe, Russo, Holandês, Sueco, Norueguês, Dinamarquês, Finlandês, Polonês, Turco, Tcheco, Húngaro, Grego, Hebraico, Hindi, Tâmil, Telugu
        
        ### 📦 **Provedores:**
        - edge-tts: provedor com vozes mais naturais (Windows/Microsoft voices). Instalar: `pip install edge-tts`
        - gTTS: rápido e simples (Google, requer internet): `pip install gTTS`
        - pyttsx3: offline (voz robótica, mas sem internet): `pip install pyttsx3`
        ''')

    # Diagnostic probe for edge-tts voices (se disponível)
    if EDGE_TTS_AVAILABLE:
        st.markdown('---')
        if st.button('🧪 Diagnosticar vozes edge-tts (probe rápido)'):
            st.info(f'Executando probe: tentarei sintetizar uma curta frase com cada voz do idioma "{selected_language}" e reportarei sucesso/erro. Isso pode demorar alguns segundos por voz.')
            probe_results = {}
            import asyncio
            try:
                import edge_tts
                async def probe_voice(v):
                    sample = 'Teste rápido de voz.'
                    tmp = f"tmp_tts_probe_{uuid.uuid4().hex[:8]}.mp3"
                    try:
                        comm = edge_tts.Communicate(sample, v)
                        await comm.save(tmp)
                        ok = os.path.exists(tmp) and os.path.getsize(tmp) > 0
                        if ok:
                            h = 'OK'
                        else:
                            h = 'EMPTY'
                        safe_remove_file(tmp)
                        return v, h
                    except Exception as e:
                        safe_remove_file(tmp)
                        return v, f'ERR:{str(e)}'

                voices_to_test = filtered_voices
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                for v in voices_to_test:
                    res = loop.run_until_complete(probe_voice(v))
                    probe_results[res[0]] = res[1]
                loop.close()
            except Exception as e:
                st.error(f'Falha ao executar probe edge-tts: {e}')
                probe_results = {'error': str(e)}

            st.subheader('Resultados do probe')
            for v, r in probe_results.items():
                st.write(f'{v}: {r}')
            st.info('Se apenas algumas vozes reportarem OK, use-as explicitamente no campo "Voz / Idioma". Se nenhuma reportar OK, verifique a instalação de edge-tts e a conectividade.')

    
    st.markdown("""
    ---
    <div style="text-align: center;">
        <p><strong>🔧 Ferramentas disponíveis: YouTube Downloader | Transcrição | SpeedTest | Removedor de Fundo</strong></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Central de Ferramentas Multimídia",
        page_icon="🚀",
        layout="wide"
    )
    
    init_session_state()
    init_transcription_session_state()

    # Menu sidebar
    st.sidebar.title("📱 Menu Principal")
    page = st.sidebar.radio(
        "Escolha uma ferramenta:", 
        [
            "🏠 Página Inicial",
            "🎬 Baixar Vídeos", 
            "🎵 Transcrever Áudio", 
            "🔊 Texto para Fala",
            "🚀 Teste de Velocidade",
            "🖼️ Remover Fundo"
        ]
    )

    # Roteamento das páginas
    if page == "🏠 Página Inicial":
        home_page()
        
    elif page == "🎬 Baixar Vídeos":
        st.title("🎬 YouTube Video Downloader")
        
        show_downloads_list()
        
        st.subheader("🔗 Baixar Novo Conteúdo")
        urls = st.text_area("Cole aqui as URLs dos vídeos, separadas por vírgula:")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎬 Baixar Vídeo Completo (MP4)"):
                url_list = [url.strip() for url in urls.split(',') if url.strip()]
                if url_list:
                    progress_bar = st.progress(0)
                    for idx, url in enumerate(url_list):
                        if not url.startswith("https://"):
                            url = "https://" + url
                        try:
                            with st.spinner(f"Processando vídeo {idx+1}/{len(url_list)}: {url}..."):
                                result = download_video(url, audio_only=False)
                                if result:
                                    add_download_to_session(result)
                                    st.success(f"✅ Vídeo {idx+1} processado: {result['title']}")
                                else:
                                    st.error(f"❌ Erro ao processar vídeo {idx+1}: {url}")
                                progress_bar.progress((idx + 1) / len(url_list))
                        except Exception as e:
                            st.error(f"❌ Erro ao processar vídeo {idx+1}: {e}")
                            progress_bar.progress((idx + 1) / len(url_list))
                    st.rerun()
                else:
                    st.warning("⚠️ Por favor, insira pelo menos uma URL válida.")

        with col2:
            if st.button("🎵 Extrair Apenas Áudio (MP3)"):
                url_list = [url.strip() for url in urls.split(',') if url.strip()]
                if url_list:
                    progress_bar = st.progress(0)
                    for idx, url in enumerate(url_list):
                        if not url.startswith("https://"):
                            url = "https://" + url
                        try:
                            with st.spinner(f"Extraindo áudio {idx+1}/{len(url_list)}: {url}..."):
                                result = download_video(url, audio_only=True)
                                if result:
                                    add_download_to_session(result)
                                    st.success(f"✅ Áudio {idx+1} extraído: {result['title']}")
                                else:
                                    st.error(f"❌ Erro ao extrair áudio {idx+1}: {url}")
                                progress_bar.progress((idx + 1) / len(url_list))
                        except Exception as e:
                            st.error(f"❌ Erro ao extrair áudio {idx+1}: {e}")
                            progress_bar.progress((idx + 1) / len(url_list))
                    st.rerun()
                else:
                    st.warning("⚠️ Por favor, insira pelo menos uma URL válida.")

    elif page == "🎵 Transcrever Áudio":
        st.title("🎵 Transcrição de Áudio/Vídeo")
        
        show_transcriptions_list()
        
        st.subheader("🎯 Nova Transcrição")
        
        # Configurações globais (fora dos expanders)
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "🌍 Idioma de Transcrição:",
                options=[
                    ("auto", "🔄 Detectar Automaticamente"),
                    ("pt-BR", "🇧🇷 Português (Brasil)"),
                    ("en-US", "🇺🇸 Inglês (EUA)"),
                    ("es-ES", "🇪🇸 Espanhol"),
                    ("fr-FR", "🇫🇷 Francês"),
                    ("de-DE", "🇩🇪 Alemão"),
                    ("it-IT", "🇮🇹 Italiano")
                ],
                format_func=lambda x: x[1],
                key="language_select"
            )
        
        with col2:
            audio_quality = st.selectbox(
                "🎵 Qualidade do Áudio:",
                options=[
                    ("standard", "📻 Padrão (16kHz, Mono)"),
                    ("high", "🎼 Alta (44kHz, Stereo)")
                ],
                format_func=lambda x: x[1],
                help="Padrão é otimizado para reconhecimento de fala. Alta qualidade para áudios musicais.",
                key="quality_select"
            )
        
        # Tabs para diferentes tipos de entrada
        tab1, tab2 = st.tabs(["📁 Enviar Arquivo", "🔗 Link do YouTube"])
        
        with tab1:
            st.markdown("### 📁 Enviar Arquivo de Áudio/Vídeo")
            uploaded_file = st.file_uploader(
                "Escolha um arquivo", 
                type=["mp4", "wav", "mp3", "m4a", "avi", "mov", "webm"],
                help="Formatos suportados: MP4, WAV, MP3, M4A, AVI, MOV, WEBM"
            )
            
            file_title = st.text_input("Título da transcrição (opcional):", 
                                     value=uploaded_file.name if uploaded_file else "",
                                     key="file_title_input")
            
            if uploaded_file is not None:
                if st.button("🎵 Iniciar Transcrição do Arquivo", key="transcribe_file"):
                    with st.spinner("Processando transcrição..."):
                        title = file_title if file_title else uploaded_file.name.split('.')[0]
                        lang_code = language[0]  # Pega o código do idioma
                        quality_code = audio_quality[0]  # Pega o código da qualidade
                        success = process_transcription("upload", uploaded_file, title, lang_code, quality_code)
                        if success:
                            st.rerun()
        
        with tab2:
            st.markdown("### 🔗 Link do YouTube")
            youtube_url = st.text_input(
                "Cole aqui o link do YouTube:", 
                placeholder="https://www.youtube.com/watch?v=...",
                key="youtube_url_input"
            )
            
            url_title = st.text_input("Título da transcrição (opcional):", 
                                    value="",
                                    key="url_title_input")
            
            if youtube_url:
                if st.button("🎵 Iniciar Transcrição do YouTube", key="transcribe_youtube"):
                    if youtube_url.startswith(("https://www.youtube.com", "https://youtu.be", "www.youtube.com", "youtu.be")):
                        if not youtube_url.startswith("https://"):
                            youtube_url = "https://" + youtube_url
                        
                        with st.spinner("Processando transcrição..."):
                            title = url_title if url_title else "Video do YouTube"
                            lang_code = language[0]  # Pega o código do idioma
                            quality_code = audio_quality[0]  # Pega o código da qualidade
                            success = process_transcription("youtube", youtube_url, title, lang_code, quality_code)
                            if success:
                                st.rerun()
                    else:
                        st.error("⚠️ Por favor, insira um link válido do YouTube")
        
        # Informações sobre o processo
        with st.expander("ℹ️ Como funciona a transcrição"):
            st.markdown("""
            **📋 Processo de Transcrição Melhorado:**
            1. **Extração de áudio** - Converte arquivo/vídeo para áudio WAV otimizado
            2. **Divisão inteligente** - Quebra em segmentos de 1.5-2 minutos
            3. **Reconhecimento múltiplo** - Tenta vários idiomas e engines:
               - 🔍 Google Speech Recognition (principal)
               - 🤖 OpenAI Whisper (se disponível)
               - 📱 CMU Sphinx (offline, fallback)
            4. **Compilação** - Junta todos os segmentos em texto final
            5. **Geração de documentos** - Cria PDF e TXT formatados
            
            **💡 Dicas para melhor resultado:**
            - ✅ **Para fala clara:** Use "Detectar Automaticamente" ou idioma específico
            - ✅ **Para áudio em inglês:** Selecione "Inglês (EUA)" 
            - ✅ **Para música/mixagens:** Use "Alta qualidade"
            - ✅ **Áudio limpo:** Evite muito ruído de fundo
            - ✅ **Velocidade normal:** Fala muito rápida pode falhar
            
            **🔧 Solução de Problemas:**
            - Se muitos chunks falharem, tente outro idioma
            - Para vídeos em inglês, mude para "en-US"
            - Áudio muito baixo: o sistema aumenta automaticamente o volume
            - Música/efeitos: pode interferir no reconhecimento de fala
            
            **📁 Armazenamento:**
            - Todas as transcrições ficam salvas na sessão atual
            - Download de PDF e TXT disponível para cada transcrição
            - Arquivos temporários são automaticamente removidos
            - Estatísticas de sucesso mostradas após processamento
            """)
            
    elif page == "🚀 Teste de Velocidade":
        run_speedtest()
    
    elif page == "🔊 Texto para Fala":
        tts_page()
        
    elif page == "🖼️ Remover Fundo":
        background_remover()

if __name__ == "__main__":
    main()
