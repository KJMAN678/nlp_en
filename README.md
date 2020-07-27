# nlp_en

#### nlp_from_archive.ipynb  
下記の記事を再現
Text Analysis & Feature Engineering with NLP  
https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d  

#### nlp_minor correction.ipynb  
上記のコードを参考に、kaggleのツイッターの自然言語処理コンペのデータをもとに自然言語処理を実施  
Real or Not? NLP with Disaster Tweets  
https://www.kaggle.com/c/nlp-getting-started/data  

## 英語の自然言語処理の特徴量の作成を学ぶ

コードを拝借しているサイト  
### Text Analysis & Feature Engineering with NLP
 https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d  

## データセット  
### News Category Dataset
 https://www.kaggle.com/rmisra/news-category-dataset
 
#### 正規表現  
```python
.str.contains(r'[^\s\w]')
```
  --> 空白、アルファベット、数字ではない記号を含む  

#### ストップワード
```python
import nltk

# 英語のストップワードの設定
nltk.download('stopwords')
lst_stopwords = nltk.corpus.stopwords.words("english")

# ストップワードの除外
txt = [word for word in txt if word not in lst_stopwords]
```

#### 語幹の週出
```python
import nltk

ps = nltk.stem.porter.PorterStemmer()
print([ps.stem(word) for word in txt])
```

#### 見出し語化
その単語を、辞書に載っている形に従って分類する  
```python
import nltk

lem = nltk.stem.wordnet.WordNetLemmatizer()
nltk.download('wordnet')
print([lem.lemmatize(word) for word in txt])
```

#### ワードクラウド
<img width="252" alt="2020-07-26_08h11_54" src="https://user-images.githubusercontent.com/45703844/88467951-b10f7880-cf17-11ea-9012-ce8c3dc8bba9.png">

```python
import wordcloud

wc = wordcloud.WordCloud(background_color='black', max_words=100, 
                         max_font_size=35)
wc = wc.generate(str(corpus))
```

#### LDAモデル

##### LDAとは

LDA は1つの文書が複数のトピックから成ることを仮定した言語モデルの一種。  
日本語だと「潜在的ディリクレ配分法」と呼ばれる。
https://bit.ly/330iXuq

##### gensim.models.ldamodel.LdaModel LDAモデル
https://fits.hatenablog.com/entry/2018/03/13/214609

```python
import gensim

lda_model = gensim.models.ldamodel.LdaModel()
```

 ### 事前準備１
 
```python
 ner = spacy.load("en_core_web_lg") 
``` 

 事前にコマンドライン上で下記を実行し、開発環境を立ち上げなおす。  
 
 ```python
 python -m spacy download en_core_web_lg
 ```
 
 このような結果が出力される。  
 <img width="706" alt="2020-07-26_07h55_04" src="https://user-images.githubusercontent.com/45703844/88467754-668cfc80-cf15-11ea-9296-912002ed54b6.png">
 
 参考サイト
 https://stackoverflow.com/questions/56470403/spacy-nlp-spacy-loaden-core-web-lg

### 事前準備２

```python
lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
```

Pythonシェルに移動し、次のように入力します。  

```python
>>> import nltk
>>> nltk.download()
```

次に、インストールウィンドウが表示される。  

<img width="434" alt="2020-07-26_08h00_56" src="https://user-images.githubusercontent.com/45703844/88467834-3db93700-cf16-11ea-9c5b-b0ead2857474.png">

[モデル]タブに移動し、[識別子]列の下から[punkt]を選択する。  

次に、[ダウンロード]をクリックすると、必要なファイルがインストールされる。  

これで完了。  

参考サイト  
https://stackoverflow.com/questions/4867197/failed-loading-english-pickle-with-nltk-data-load

