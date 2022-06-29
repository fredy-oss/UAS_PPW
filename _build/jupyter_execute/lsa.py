#!/usr/bin/env python
# coding: utf-8

# # Crawling Data

# <p>Data crawling adalah program yang menghubungkan halaman web, kemudian mengunduh kontennya. Program crawling dalam data science hanya akan online untuk mencari dua hal, yaitu data yang dicari oleh pengguna dan penjelajahan target dengan jangkauan yang lebih luas.</p> 

# 

# <h1>Install Library nltk</h1>

# In[1]:


pip install nltk


# <h1>Import Library nltk</h1>

# In[2]:


import nltk
nltk.download('stopwords')


# <h1>Import Library</h1>

# In[3]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# <h1>Install Library sklearn</h1>

# In[4]:


pip install sklearn


# <h1>Install Library pandas</h1>

# In[5]:


pip install pandas


# <h1>Baca Data PTA</h1>

# In[6]:


df = pd.read_csv('./crawlingpta.csv')


# In[77]:


df.head(10)


# In[78]:


df.drop(['Judul'],axis=1,inplace=True)
df.drop(['Penulis'],axis=1,inplace=True)
df.drop(['Dospem 1'],axis=1,inplace=True)
df.drop(['Dospem 2'],axis=1,inplace=True)
df.drop(['Abstraction'],axis=1,inplace=True)
df.drop(['Link Download'],axis=1,inplace=True)


# In[79]:


df.head(10)


# In[80]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[81]:


vect_text=vect.fit_transform(df['Abstraksi'])


# In[82]:


print(vect_text.shape)
#print(vect_text)
type(vect_text)
df = pd.DataFrame(vect_text.toarray())
print(df)
idf=vect.idf_


# In[83]:


idf=vect.idf_


# In[84]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['dan'])
print(dd['cumi'])  # police is most common and forecast is least common among the news headlines.


# <h1>Topik modeling menggunakan LDA dan LSA</h1>

# <p>LDA (Linear Discriminant Analysis) adalah teknik statistika klasik yang sudah dipakai sejak lama untuk mereduksi dimensi. Dengan LDA, kita juga bisa melakukan pembagian data ke dalam beberapa kelompok (clustering). </p>

# <p>Latent Semantic Analysis (LSA) merupakan sebuah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi.</p>

# <h1>Algoritma LSA</h1>

# <p>Tahapan-tahapan algoritma LSA dalam prosessing teks</p>

# <h1>Menghitung Term-document Matrix</h1> 

# <p>Document Term Matrix merupakan algoritma – Metode perhitungan yang sering kita temui dalam text minning.</p>

# <p>Melalui Document Term Matrix, kita dapat melakukan analisis yang lebih menarik. Mudah untuk menentukan jumlah kata individual untuk setiap dokumen atau untuk semua dokumen. Misalkan untuk menghitung agregat dan statistik dasar seperti jumlah istilah rata-rata, mean, median, mode, varians, dan deviasi standar dari panjang dokumen, serta dapat mengetahui istilah mana yang lebih sering dalam kumpulan dokumen dan dapat menggunakan informasi tersebut untuk menentukan istilah mana yang lebih mungkin “mewakili” dokumen tersebut.</p>

# <h1>Singular Value Decomposition</h1>

# <p>Singular Value Decomposition adalah seuatu teknik untuk mendekomposisi matriks berukuran apa saja (biasanya diaplikasikan untuk matriks dengan ukuran sangat besar), untuk mempermudah pengolahan data. Hasil dari SVD ini adalah singular value yang disimpan dalam sebuah matriks diagonal, D,  dalam urutan yang sesuai dengan koresponding singular vector-ya.</p>

# In[85]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[86]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[87]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[88]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# <h1>Mengekstrak Topik dan Term</h1>

# In[89]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:




