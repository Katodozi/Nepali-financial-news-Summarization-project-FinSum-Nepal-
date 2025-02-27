FinSum Nepal is the Nepali financial text summarization project where we are using Natural language Processing (NLP), specifically nltk for text preprocessing.
Among abstractive and extractive summarization methods to perform the summarization task,
in this project we are using extractve summarization.

1. Web scrapping(check the repository named Web-Scraping)
   (Fully cleaned data is uploaded on kaggle in the link given below)
    https://www.kaggle.com/datasets/anujbhattrai/the-nepali-financial-news-dataset
2. Text preprocessing(Total cleaning of the scrapped datasets was done in this step and then the nlp text preprocessing too(stopword removal, stemming manually)
3. Word embedding(Conversion of the tokenized and stemmed words into the vector values of 100 dimension using word2vec and building our own custom model.)
4. Extractive summarization(the textrank summarization algorithm integrated with word embedding as cosine similarity for making the summary simantically rich)
5. Django app directory(Finally converting the summarizer system that we have build into the web applications)
