import fitz
import nltk
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import string

filepath = "/Users/babajaguska/Downloads/Resume Minja Belic.pdf"
with fitz.open(filepath) as pdf:
    text = ''
    for page_num in range(1):
        page = pdf[page_num]
        text += page.get_text()



print(text[:1000])
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

text = text.translate(str.maketrans('', '', string.punctuation))
text = text.lower()

wordcloud = WordCloud(width = 900, height = 800,
                background_color ='white',
                # stopwords = stop_words,
                min_font_size = 10).generate(text)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.savefig("wordcloud_cv.png")