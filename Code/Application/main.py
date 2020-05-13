import tkinter as tk
from tkinter import *
from tkinter import filedialog
# import tkinter.scrolledtext as tkst
from tkinter.messagebox import *
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from bs4 import BeautifulSoup
import re
import nltk
from nltk import FreqDist
import spacy
import gensim
from gensim import corpora
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

import joblib
from textblob import TextBlob



class PrePage(tk.Toplevel):

    def __init__(self):
        super().__init__()
        self.title('Prepare for the application')
        # self.geometry('%d%d' % (300, 200))
        self.DataFile_Path = StringVar()
        self.createPage()

    def getData(self):
        DataFile_path = filedialog.askopenfilename()
        self.DataFile_Path.set(DataFile_path)

    def confirm_file(self):
        DataFilePath = self.DataFile_Path.get()

        if DataFilePath != '':
            self.dataPath = [DataFilePath]
            self.destroy()
        else:
            showinfo(title='error!', message='Please load a real file first')

    def cancel(self):
        self.dataPath = None
        self.destroy()

    def createPage(self):
        self.page = Frame(self)
        self.page.pack()
        Label(self.page).grid(row=0, stick=W)
        Button(self.page, text='Load Model: ', command=self.getData).grid(row=1, stick=W, pady=10)
        Entry(self.page, textvariable=self.DataFile_Path).grid(row=1, column=1, stick=E)
        Button(self.page, text='Confirm', command=self.confirm_file).grid(row=2, stick=W, pady=10)
        Button(self.page, text='Cancel', command=self.cancel).grid(row=2, column=1, stick=E)

class MainPage(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MovieGrade')
        # self.geometry('%d*%d+%d+%d' % (600, 320,10,10))
        self.review = StringVar()
        self.prediction = StringVar()
        self.score = StringVar()
        self.topics = StringVar()
        self.createPage()

    def createNewPage(self):
        FilePath = PrePage()
        self.wait_window(FilePath)
        return FilePath.dataPath

    def loadModel(self):
        review = self.review.get()
        if review == '':
            showinfo('Error!', 'Please input a review first')
        else:
            res = self.createNewPage()
            if res is None:
                return
            dataPath = res[0]

            return dataPath

    def review_to_text(self, data, remove_stopwords):

        sentences = nltk.tokenize.sent_tokenize(data)

        wordList = []

        for sentence in sentences:
            letters = re.sub('[^a-zA-Z]', ' ', sentence)
            words = letters.lower().split()

            if remove_stopwords:
                all_stop_words = list(stopwords.words('english')) + ['movie', 'film', 'good', 'great', 'review', 'just', 'like', 'enjoy', 'best',
                   'wa', 'hi', 'ha', 'movi']
                words = [w for w in words if w not in all_stop_words]
                words = [w for w in words if len(w) > 2]
                wordList.append(words)

        return wordList

    def showResult(self):

        # review = [' '.join(self.review_to_text(self.review.get(), True))]
        review = self.review_to_text(self.review.get(), True)

        reviews = []
        for rev1 in review:
            # rev = [' '.join(word for word in rev1)]
            rev = ' '.join(rev1)
            reviews.append(rev)

        review_for_model = [' '.join(reviews)]


        # print(review_for_model)


        dataPath = self.loadModel()
        model = joblib.load(dataPath)
        pred = model.predict(review_for_model)

        # sentences = nltk.tokenize.sent_tokenize(self.review.get())

        # tmp = [' '.join(self.review_to_text(sentences, True))]
        nlp = spacy.load('en', disable=['parser', 'ner'])

        review_for_topic = []
        tags = ['NOUN', 'ADJ']
        for sent in review:
            doc = nlp(" ".join(sent))
            review_for_topic.append([token.lemma_ for token in doc if token.pos_ in tags])

        dictionary = corpora.Dictionary(review_for_topic)
        doc_term_matrix = [dictionary.doc2bow(review) for review in review_for_topic]

        LDA = gensim.models.ldamodel.LdaModel

        lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                        chunksize=1000, passes=50)

        topicList = lda_model.print_topics(num_topics=1, num_words=3)
        topicStrings = topicList[0][1].split('"')
        topStr = topicStrings[1] + ', ' + topicStrings[3] + ', ' + topicStrings[5]

        blobReview = TextBlob(self.review.get())
        score = round(blobReview.sentiment.polarity * 2.5 + 2.5)

        if pred == 1:
            res = 'Positive'
        else:
            res = 'Negative'

        self.topics.set(topStr)
        self.prediction.set(res)
        self.score.set(str(score))

    def cleanResult(self):
        self.score.set("")
        self.prediction.set("")
        self.topics.set("")

    def createPage(self):
        self.page = Frame(self)
        self.page.pack()
        Label(self.page).grid(row = 0, stick = W)
        Label(self.page, text = 'Input Review here').grid(row = 1, stick = W, pady = 10)
        Entry(self.page, textvariable = self.review).grid(row = 1, column = 1, stick = E)
        Label(self.page, text = 'Topic').grid(row = 2, stick = W, pady = 10)
        Label(self.page, textvariable = self.topics, relief = 'ridge', width = 21, height = 1).grid(row = 2, column = 1, stick = E)
        Label(self.page, text = 'P/N').grid(row = 3, stick = W, pady = 10)
        Label(self.page, textvariable = self.prediction, relief = 'ridge', width = 21, height = 1).grid(row = 3, column = 1, stick = E)
        Label(self.page, text = 'score').grid(row = 4, stick = W, pady = 10)
        Label(self.page, textvariable = self.score, relief  = 'ridge', width = 21, height = 1).grid(row = 4, column = 1, stick = E)
        Button(self.page, text = 'Show Result', command = self.showResult).grid(row = 5, pady = 10)
        Button(self.page, text = 'Clean Result', command = self.cleanResult).grid(row = 5, column = 1, pady = 10)
        Button(self.page, text = 'Exit', command = self.page.quit).grid(row = 5, stick = E, column = 2, pady = 10)



if __name__ == '__main__':
    app = MainPage()
    app.mainloop()