import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Certifique-se de ter baixado os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Função para extrair texto do PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
            return text
    except FileNotFoundError:
        return "Arquivo não encontrado. Verifique o caminho."

# Função para pré-processar o texto
def preprocess_text(text):
    stop_words = set(stopwords.words('portuguese'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

# Função para gerar nuvem de palavras
def generate_wordcloud(words):
    wordcloud = WordCloud(width=1300, height=600, background_color='black').generate(' '.join(words))
    plt.figure(figsize=(13, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Função para análise de sentimentos
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Caminho do arquivo PDF
file_path = '/storage/emulated/0/Download/soniapaiva,+5º+-+50333-Texto+do+artigo-209964-1-4-20190917+-+DIAGRAMADO+(2).pdf'

# Extrair texto do PDF
pdf_text = extract_text_from_pdf(file_path)
if pdf_text != "Arquivo não encontrado. Verifique o caminho.":
    
    # Pré-processar o texto
    filtered_words = preprocess_text(pdf_text)
    
    # Gerar nuvem de palavras
    generate_wordcloud(filtered_words)
    
    # Analisar sentimentos
    sentiment = sentiment_analysis(pdf_text)
    print(f"Resultado da análise de sentimentos: {sentiment}")
else:
    print(pdf_text)