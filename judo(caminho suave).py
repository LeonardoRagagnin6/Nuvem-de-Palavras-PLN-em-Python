import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from PIL import Image, ImageEnhance
import numpy as np

# Baixar stopwords (se necessário)
nltk.download('stopwords')

# Passo 1: Raspar o conteúdo do site
url = "https://www.judosamuraikan.com.br/2019/06/guia-dos-principais-golpes-usados.html?m=1"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Passo 2: Extrair o texto da página
text = soup.get_text()

# Passo 3: Pré-processamento do texto (remover stopwords)
stop_words = set(stopwords.words('portuguese'))
words = text.split()

# Filtrando palavras sem importância (stopwords)
filtered_words = [word for word in words if word.lower() not in stop_words]

# Juntar as palavras filtradas
filtered_text = " ".join(filtered_words)

# Passo 4: Carregar a imagem (que será a máscara)
mask_image_path = "/storage/emulated/0/Download/kata.jpeg"  # Certifique-se de que este caminho está correto
try:
    # Carregar a imagem como máscara
    mask_image = Image.open(mask_image_path).convert('L')  # Convertendo para tons de cinza
    
    # Ajustar o contraste para intensificar as áreas escuras
    enhancer = ImageEnhance.Contrast(mask_image)
    mask_image = enhancer.enhance(2.0)  # Aumenta o contraste; ajuste conforme necessário

    # Redimensionar a imagem para 800x800 pixels
    mask_image = mask_image.resize((800, 800))  # Ajuste o tamanho conforme necessário

    # Converter a imagem para um array NumPy
    mask_array = np.array(mask_image)

    # Inverter a máscara (preto fica como 255, branco fica como 0)
    mask_array = np.invert(mask_array)
    
except FileNotFoundError:
    print(f"Erro: Arquivo {mask_image_path} não encontrado.")
    mask_array = None  # Para continuar sem a máscara caso ocorra erro

# Passo 5: Criar a nuvem de palavras com a máscara
wordcloud = WordCloud(
    width=800,
    height=800,
    background_color="white",
    mask=mask_array,
    contour_width=1,
    contour_color='black',
    max_font_size=100,  # Aumentar o tamanho máximo da fonte
    scale=2,  # Aumentar a escala para palavras maiores
    colormap='tab20',  # Usar uma coloração com múltiplas cores
).generate(filtered_text)

# Passo 6: Exibir a nuvem de palavras com a máscara
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()