import streamlit as st
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import time
import numpy as np
from rembg import remove
import streamlit.components.v1 as components
import json

# Configuração da página
st.set_page_config(
    page_title="🎨 Removedor de Fundo Avançado",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #2E86AB; margin-bottom: 2rem; }
    .stAlert { border-radius: 10px; }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }
    .canvas-section {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
<div class='upload-section'>
    <h1 style='margin: 0; color: white; text-align: center;'>🎨 Removedor de Fundo Avançado</h1>
    <p style='margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1rem; text-align: center;'>
        Remoção automática de fundo com IA + edição manual opcional
    </p>
</div>
""", unsafe_allow_html=True)

# Helper para converter PIL para bytes
def pil_to_bytes(img: Image.Image, fmt='PNG'):
    from io import BytesIO
    buf = BytesIO()
    try:
        if fmt.upper() == 'JPEG':
            rgb = img.convert('RGB')
            rgb.save(buf, format='JPEG', quality=95)
        else:
            img.save(buf, format=fmt)
    except Exception:
        img.convert('RGBA').save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()

def pil_to_base64(img: Image.Image, fmt='PNG'):
    """Converte imagem PIL para string base64 para uso em HTML"""
    import base64
    img_bytes = pil_to_bytes(img, fmt)
    return base64.b64encode(img_bytes).decode('utf-8')

def render_modern_gallery(original, processed, mask_img=None):
    """Renderiza galeria moderna com cards e downloads"""
    
    st.markdown("### 🖼️ **Resultados do Processamento**")
    
    images_data = []
    
    if original is not None:
        images_data.append({
            'title': 'Original',
            'image': original,
            'icon': '📷',
            'color': '#6c757d'
        })
        
    if processed is not None:
        images_data.append({
            'title': 'Sem Fundo',
            'image': processed,
            'icon': '✨',
            'color': '#28a745'
        })
        
    if mask_img is not None:
        images_data.append({
            'title': 'Máscara',
            'image': mask_img,
            'icon': '🎭',
            'color': '#6f42c1'
        })

    if not images_data:
        st.info('🤷‍♂️ Nenhuma imagem processada ainda.')
        return

    # Renderizar em grid responsivo
    cols = st.columns(len(images_data))
    
    for i, img_data in enumerate(images_data):
        with cols[i]:
            render_image_card(img_data)

def render_image_card(img_data):
    """Renderiza card individual de imagem"""
    
    img = img_data['image']
    title = img_data['title']
    icon = img_data['icon']
    color = img_data['color']
    
    # Card container
    st.markdown(f"""
    <div style='background: white; border-radius: 12px; padding: 15px; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid {color}; margin-bottom: 20px;'>
        <h4 style='margin: 0 0 10px 0; color: {color}; text-align: center;'>
            {icon} {title}
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Exibir imagem com fundo transparente para PNGs
    if img.mode == 'RGBA':
        # Criar fundo xadrez para mostrar transparência
        st.markdown("""
        <div style='background: linear-gradient(45deg, #f0f0f0 25%, transparent 25%), 
        linear-gradient(-45deg, #f0f0f0 25%, transparent 25%), 
        linear-gradient(45deg, transparent 75%, #f0f0f0 75%), 
        linear-gradient(-45deg, transparent 75%, #f0f0f0 75%); 
        background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0px; 
        padding: 10px; border-radius: 8px;'>
        """, unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.image(img, use_container_width=True)
    
    # Informações da imagem
    width, height = img.size
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 10px; border-radius: 6px; margin: 10px 0;'>
        <small>
            <strong>📏 Dimensões:</strong> {width} × {height} px<br>
            <strong>🎨 Modo:</strong> {img.mode}<br>
            <strong>🌈 Transparência:</strong> {'Sim' if img.mode == 'RGBA' else 'Não'}
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    # Botão de download
    try:
        fmt = 'PNG' if img.mode == 'RGBA' else 'JPEG'
        full_bytes = pil_to_bytes(img, fmt=fmt)
        filename = f"{title.replace(' ', '_')}.{fmt.lower()}"
        
        st.download_button(
            label=f"📥 Baixar {title}",
            data=full_bytes,
            file_name=filename,
            mime=f"image/{fmt.lower()}",
            type="primary" if "Sem Fundo" in title else "secondary",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Erro ao preparar download: {e}")

def process_with_rembg(input_image):
    """Processa remoção de fundo com rembg"""
    
    try:
        from rembg import new_session, remove
        import io
        
        # Criar sessão
        session = new_session('u2net')
        
        # Converter PIL para bytes
        img_byte_array = io.BytesIO()
        input_image.save(img_byte_array, format='PNG')
        img_bytes = img_byte_array.getvalue()
        
        # Processar remoção de fundo
        result_bytes = remove(img_bytes, session=session)
        
        # Converter resultado de volta para PIL
        result_image = Image.open(io.BytesIO(result_bytes))
        
        return result_image
        
    except Exception as e:
        st.error(f"❌ Erro com rembg: {e}")
        return None

def apply_brush_to_image(input_image, brush_points, edit_mode):
    """Aplica os dados do pincel à imagem para criar edição manual"""
    try:
        # Criar uma máscara baseada nos pontos do pincel
        mask = Image.new('L', input_image.size, 255)  # Máscara branca = manter por padrão
        draw = ImageDraw.Draw(mask)

        # Aplicar cada ponto do pincel
        for point in brush_points:
            x, y = point['x'], point['y']
            brush_size = point['size']
            brush_color = point['color']
            opacity = point['opacity']

            # Calcular raio baseado no tamanho do pincel
            radius = brush_size // 2

            # Criar coordenadas do círculo
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(input_image.size[0], x + radius)
            y2 = min(input_image.size[1], y + radius)

            # Desenhar círculo na máscara
            # Preto (0) = área a ser editada, Branco (255) = área a manter
            color_value = 0 if 'remover' in brush_color.lower() else 255
            draw.ellipse([x1, y1, x2, y2], fill=color_value)

        # Aplicar a máscara à imagem
        if input_image.mode != 'RGBA':
            input_image = input_image.convert('RGBA')

        # Criar imagem de resultado
        result_image = input_image.copy()

        # Para "Remover áreas específicas": tornar áreas pintadas transparentes
        if edit_mode == "Remover áreas específicas":
            # Inverter a máscara - onde pintamos (0) deve ficar transparente
            inverted_mask = Image.new('L', mask.size, 255)
            inverted_draw = ImageDraw.Draw(inverted_mask)

            for x in range(mask.size[0]):
                for y in range(mask.size[1]):
                    if mask.getpixel((x, y)) == 0:  # Pintado para remover
                        inverted_mask.putpixel((x, y), 0)  # Transparente

            result_image.putalpha(inverted_mask)

        # Para "Manter apenas área selecionada": manter apenas áreas pintadas
        elif edit_mode == "Manter apenas área selecionada":
            # Onde pintamos (255 na máscara) deve ficar visível, resto transparente
            transparent_mask = Image.new('L', mask.size, 0)
            transparent_draw = ImageDraw.Draw(transparent_mask)

            for x in range(mask.size[0]):
                for y in range(mask.size[1]):
                    if mask.getpixel((x, y)) == 255:  # Pintado para manter
                        transparent_draw.point((x, y), fill=255)  # Manter

            result_image.putalpha(transparent_mask)

        return result_image

    except Exception as e:
        st.error(f"Erro ao aplicar pincel: {str(e)}")
        return input_image.copy()

def process_with_canvas_mask(input_image, mask_image):
    """Aplica máscara criada no canvas"""
    
    if mask_image is None:
        return input_image
    
    try:
        # Converter imagens para RGBA
        img_rgba = input_image.convert('RGBA')
        
        # Redimensionar máscara para coincidir com a imagem
        mask_resized = mask_image.resize(img_rgba.size, Image.Resampling.NEAREST)
        mask_array = np.array(mask_resized)
        
        # Criar máscara binária (onde foi desenhado = transparent, resto = opaque)
        # Canvas desenha em branco onde marcamos para remover
        binary_mask = (mask_array[:, :, 0] > 128)  # Detectar áreas brancas (desenhadas)
        
        # Aplicar máscara
        img_array = np.array(img_rgba)
        img_array[binary_mask, 3] = 0  # Tornar transparente onde foi marcado
        
        result = Image.fromarray(img_array, 'RGBA')
        return result
        
    except Exception as e:
        st.error(f"❌ Erro ao aplicar máscara: {e}")
        return input_image

# Sidebar com opções
st.sidebar.title("🎛️ Opções de Remoção")

removal_method = st.sidebar.radio(
    "Escolha o método:",
    ["🤖 IA Automática (rembg)"]
)

# Interface principal
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📤 **Upload de Imagem**")
    
    uploaded_file = st.file_uploader(
        "Escolha uma imagem", 
        type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
        help="Formatos suportados: PNG, JPG, JPEG, WEBP, BMP"
    )

with col2:
    if uploaded_file is not None:
        st.markdown("## 🖼️ **Prévia**")
        input_image = Image.open(uploaded_file)
        st.image(input_image, use_container_width=True)
        
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <strong>📁 Nome:</strong> {uploaded_file.name}<br>
            <strong>📏 Dimensões:</strong> {input_image.size[0]} × {input_image.size[1]} px<br>
            <strong>🎨 Modo:</strong> {input_image.mode}<br>
            <strong>📦 Tamanho:</strong> {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

# Processamento
if uploaded_file is not None:
    st.markdown("---")

    # Edição manual opcional
    st.markdown("### 🖌️ **Edição Manual (Opcional)**")

    edit_mode = st.radio(
        "Modo de edição:",
        ["Nenhuma edição", "Remover áreas específicas", "Manter apenas área selecionada"],
        help="Escolha como deseja editar a imagem manualmente"
    )

    if edit_mode != "Nenhuma edição":
        st.markdown("#### 🎯 **Pincel de Edição Manual:**")

        # Inicializar pontos do pincel no session_state
        if 'brush_points' not in st.session_state:
            st.session_state.brush_points = []

        # Configurações do pincel
        col1, col2, col3 = st.columns(3)
        with col1:
            brush_size = st.slider("Tamanho do pincel", 5, 50, 20)
        with col2:
            brush_color = st.selectbox("Cor do pincel", ["Preto (remover)", "Branco (manter)"], index=0)
        with col3:
            brush_opacity = st.slider("Opacidade", 0.1, 1.0, 1.0)

        # Mostrar imagem interativa para seleção de pontos
        st.markdown("#### 🖼️ **Clique na imagem para adicionar pontos do pincel:**")

        # Criar imagem de visualização com pontos
        viz_image = input_image.copy()
        if viz_image.mode != 'RGBA':
            viz_image = viz_image.convert('RGBA')
        draw_viz = ImageDraw.Draw(viz_image)

        # Desenhar pontos existentes
        for point in st.session_state.brush_points:
            x, y = point['x'], point['y']
            radius = point['size'] // 2
            color = (255, 0, 0, 128) if 'remover' in point['color'].lower() else (0, 255, 0, 128)
            draw_viz.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
            # Centro do ponto
            draw_viz.ellipse([x-3, y-3, x+3, y+3], fill=(255, 255, 255, 255))

        # Interface para mover o pincel em quatro direções
        st.markdown("**🎯 Mover Pincel - Quatro Direções:**")

        # Inicializar posição do pincel se não existir
        if 'brush_position' not in st.session_state:
            st.session_state.brush_position = {
                'x': input_image.size[0] // 2,
                'y': input_image.size[1] // 2
            }

        # Controles para mover o pincel
        col_up, col_left, col_center, col_right, col_down = st.columns(5)

        with col_center:
            st.markdown("**Posição Atual:**")
            st.markdown(f"**X:** {st.session_state.brush_position['x']} | **Y:** {st.session_state.brush_position['y']}")

        with col_up:
            if st.button("⬆️ **Cima**", use_container_width=True):
                st.session_state.brush_position['y'] = max(0, st.session_state.brush_position['y'] - 10)
                st.rerun()

        with col_left:
            if st.button("⬅️ **Esquerda**", use_container_width=True):
                st.session_state.brush_position['x'] = max(0, st.session_state.brush_position['x'] - 10)
                st.rerun()

        with col_right:
            if st.button("➡️ **Direita**", use_container_width=True):
                st.session_state.brush_position['x'] = min(input_image.size[0] - 1, st.session_state.brush_position['x'] + 10)
                st.rerun()

        with col_down:
            if st.button("⬇️ **Baixo**", use_container_width=True):
                st.session_state.brush_position['y'] = min(input_image.size[1] - 1, st.session_state.brush_position['y'] + 10)
                st.rerun()

        # Controles de movimento fino
        st.markdown("**🎚️ Movimento Fino:**")
        col_fine1, col_fine2 = st.columns(2)
        with col_fine1:
            x_adjust = st.slider("Ajuste X", -50, 50, 0, key="x_adjust")
        with col_fine2:
            y_adjust = st.slider("Ajuste Y", -50, 50, 0, key="y_adjust")

        # Aplicar ajustes finos
        current_x = st.session_state.brush_position['x'] + x_adjust
        current_y = st.session_state.brush_position['y'] + y_adjust

        # Limitar aos limites da imagem
        current_x = max(0, min(current_x, input_image.size[0] - 1))
        current_y = max(0, min(current_y, input_image.size[1] - 1))

        # Botão para adicionar ponto na posição atual
        if st.button("🎨 **Adicionar Ponto Aqui**", type="primary", use_container_width=True):
            point = {
                'x': current_x,
                'y': current_y,
                'size': brush_size,
                'color': brush_color,
                'opacity': brush_opacity
            }
            st.session_state.brush_points.append(point)
            st.success(f"✅ Ponto adicionado em ({{current_x}}, {{current_y}})")
            st.rerun()# Mostrar pontos atuais com visualização
        if st.session_state.brush_points:
            st.markdown("**📋 Pontos Marcados:**")

            # Criar visualização da imagem com pontos marcados
            viz_image = input_image.copy()
            if viz_image.mode != 'RGBA':
                viz_image = viz_image.convert('RGBA')
            draw_viz = ImageDraw.Draw(viz_image)

            points_df = []
            for i, point in enumerate(st.session_state.brush_points):
                points_df.append({
                    'Ponto': i+1,
                    'X': point['x'],
                    'Y': point['y'],
                    'Tamanho': point['size'],
                    'Cor': point['color'],
                    'Opacidade': point['opacity']
                })

                # Desenhar ponto na visualização
                x, y = point['x'], point['y']
                radius = point['size'] // 2
                color = (255, 0, 0, 128) if 'remover' in point['color'].lower() else (0, 255, 0, 128)  # Vermelho para remover, verde para manter

                # Desenhar círculo semi-transparente
                draw_viz.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

                # Desenhar centro do ponto
                draw_viz.ellipse([x-3, y-3, x+3, y+3], fill=(255, 255, 255, 255))

            # Mostrar imagem com pontos
            st.image(viz_image, caption="Visualização dos pontos marcados (🔴 = remover, 🟢 = manter)", use_container_width=True)

            # Mostrar tabela de pontos
            import pandas as pd
            st.dataframe(pd.DataFrame(points_df), use_container_width=True)

            # Botões de controle
            col_clear, col_apply = st.columns(2)
            with col_clear:
                if st.button("🗑️ **Limpar Todos os Pontos**", use_container_width=True):
                    st.session_state.brush_points = []
                    st.success("✅ Todos os pontos foram removidos!")

            with col_apply:
                if st.button("✏️ **Aplicar Edição com Pincel**", type="primary", use_container_width=True):
                    if not st.session_state.brush_points:
                        st.error("❌ Adicione pelo menos um ponto antes de aplicar!")
                    else:
                        with st.spinner("Aplicando pincel..."):
                            # Aplicar edição baseada nos pontos
                            edited_image = apply_brush_to_image(input_image, st.session_state.brush_points, edit_mode)

                        st.success("✅ Edição com pincel aplicada!")
                        st.image(edited_image, caption="Imagem após edição com pincel", use_container_width=True)

                        # Download
                        edited_bytes = pil_to_bytes(edited_image)
                        st.download_button(
                            label="💾 **Baixar Imagem Editada**",
                            data=edited_bytes,
                            file_name=f"pincel_{uploaded_file.name}",
                            mime="image/png",
                            key="download_brushed"
                        )
        else:
            st.info("ℹ️ Nenhum ponto marcado ainda. Adicione pontos clicando em 'Adicionar Ponto'.")

        # Instruções
        with st.expander("📖 **Como usar o Pincel**"):
            st.markdown("""
            ### 🎯 **Modos de Edição:**
            - **"Remover áreas específicas"**: As áreas pintadas com "Preto (remover)" ficarão transparentes
            - **"Manter apenas área selecionada"**: Apenas as áreas pintadas com "Branco (manter)" ficarão visíveis

            ### 🖌️ **Como adicionar pontos:**
            1. **Configure o pincel**: Ajuste tamanho, cor e opacidade
            2. **Digite as coordenadas**: Use os campos X e Y para especificar a posição
            3. **Adicione o ponto**: Clique em "Adicionar" para colocar o ponto
            4. **Visualize**: Veja os pontos marcados na imagem (🔴 = remover, 🟢 = manter)
            5. **Aplique a edição**: Clique em "Aplicar Edição com Pincel"

            ### 🎨 **Cores do pincel:**
            - **Preto (remover)**: Remove a área ao redor do ponto
            - **Branco (manter)**: Mantém a área ao redor do ponto

            ### 💡 **Dicas:**
            - Use coordenadas aproximadas primeiro, depois ajuste
            - Pontos maiores afetam áreas maiores
            - Use múltiplos pontos para edições complexas
            - A visualização mostra exatamente onde os pontos estão
            """)


# Processamento automático
    st.markdown("---")
    st.markdown("### 🤖 **Processamento Automático**")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 **Processar com IA**", type="primary", use_container_width=True):
                with st.spinner("🤖 Processando com IA..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.info("🔍 Carregando modelo...")
                    progress_bar.progress(20)
                    
                    status_text.info("🎯 Analisando imagem...")
                    progress_bar.progress(50)
                    
                    result_rembg = process_with_rembg(input_image)
                    
                    status_text.info("✅ Finalizando...")
                    progress_bar.progress(100)
                    
                    if result_rembg:
                        status_text.success("✅ Processamento concluído!")
                        st.markdown("---")
                        render_modern_gallery(input_image, result_rembg)
                    else:
                        status_text.error("❌ Falha no processamento")

else:
    # Estado vazio - mostrar dicas
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
        padding: 20px; border-radius: 12px; margin-bottom: 15px;'>
            <h4 style='margin: 0 0 10px 0; color: #2c3e50;'>🤖 Remoção Automática</h4>
            <ul style='margin: 0; color: #34495e;'>
                <li>Powered by rembg + IA</li>
                <li>Ideal para pessoas e objetos</li>
                <li>Processamento rápido</li>
                <li>Sem necessidade de edição</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
        padding: 20px; border-radius: 12px; margin-bottom: 15px;'>
            <h4 style='margin: 0 0 10px 0; color: #2c3e50;'>🖌️ Canvas Manual</h4>
            <ul style='margin: 0; color: #34495e;'>
                <li>Controle total sobre remoção</li>
                <li>Ideal para ajustes finos</li>
                <li>Desenhe para marcar áreas</li>
                <li>Perfeito para detalhes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        🎨 Removedor de Fundo Avançado • 
        <span style='color: #667eea;'>IA + Canvas</span> • 
        <span style='color: #ff6b6b;'>❤️</span> Feito com Streamlit
    </p>
</div>
""", unsafe_allow_html=True)