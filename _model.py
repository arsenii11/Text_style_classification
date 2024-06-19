import json
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
import joblib
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Загрузка стоп-слов
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class TextModel:
    def __init__(self):
        self.vectorizer = None
        self.model = None

        # Вызов функций для получения текстов различных стилей
        self.literary_texts = self.get_literary_texts()
        self.scientific_texts = self.get_scientific_texts()
        self.journalistic_texts = self.get_journalistic_texts()
        self.official_texts = self.get_official_texts()

        # self.literary_texts_x = [item for sublist in self.literary_texts for item in sublist]
        # self.scientific_texts_x = [item for sublist in self.scientific_texts for item in sublist]
        # self.journalistic_texts_x = [item for sublist in self.journalistic_texts for item in sublist]
        # self.official_texts_x = [item for sublist in self.official_texts for item in sublist]

        self.literary_texts_x = "\n".join(self.literary_texts).split('\r\n\r\n')
        self.scientific_texts_x = self.scientific_texts
        self.journalistic_texts_x = self.journalistic_texts
        self.official_texts_x = "\n".join(self.official_texts).split('\r\n\r\n')

        print("Literary: ", len(self.literary_texts))
        print("Scientific: ", len(self.scientific_texts))
        print("Journalistic: ", len(self.journalistic_texts))
        print("Official: ", len(self.official_texts), "\n\n")
        print("Literary X: ", len(self.literary_texts_x))
        print("Scientific X: ", len(self.scientific_texts_x))
        print("Journalistic X: ", len(self.journalistic_texts_x))
        print("Official X: ", len(self.official_texts_x))

        self.texts = self.literary_texts_x + self.scientific_texts_x + self.journalistic_texts_x + self.official_texts_x
        self.labels = ['literary'] * len(self.literary_texts_x) + ['scientific'] * len(self.scientific_texts_x) + ['journalistic'] * len(
            self.journalistic_texts_x) + ['official'] * len(self.official_texts_x)

        # Пример вывода собранных текстов
        # print("Literary Texts:", literary_texts[0])
        # print("Scientific Texts:", scientific_texts[0])
        # print("Journalistic Texts:", journalistic_texts[0])
        # print("Official Texts:", official_texts[0])

    # Функция для загрузки художественных текстов из Project Gutenberg
    def get_literary_texts(self):
        with open('literary_texts.json', 'r') as f:
            literary_texts_tmp = json.load(f)

        if len(literary_texts_tmp) > 0:
            return literary_texts_tmp

        # Загрузим несколько книг (например, "War and Peace", "Pride and Prejudice")
        urls = [
            'https://www.gutenberg.org/files/2600/2600-0.txt',  # War and Peace by Leo Tolstoy
            'https://www.gutenberg.org/files/2554/2554-0.txt',  # Crime and Punishment by Fyodor Dostoevsky
            'https://www.gutenberg.org/files/600/600-0.txt',    # Notes from the Underground by Fyodor Dostoevsky
            'https://www.gutenberg.org/files/1342/1342-0.txt'   # Pride and Prejudice by Jane Austen
        ]
        for url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                literary_texts_tmp.append(response.text)

        with open('literary_texts.json', 'w') as f:
            json.dump(literary_texts_tmp, f, indent=4)

        return literary_texts_tmp

    # Функция для загрузки научных текстов из arXiv
    def get_scientific_texts(self):
        with open('scientific_texts.json', 'r') as f:
            scientific_texts_tmp = json.load(f)

        if len(scientific_texts_tmp) > 0:
            return scientific_texts_tmp

        query = 'cat:cs.CL'  # Компьютерная лингвистика как пример
        url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=10000'
        response = requests.get(url)
        if response.status_code == 200:
            entries = response.text.split('<entry>')
            for entry in entries[1:]:
                summary = entry.split('<summary>')[1].split('</summary>')[0]
                scientific_texts_tmp.append(summary)

        with open('scientific_texts.json', 'w') as f:
            json.dump(scientific_texts_tmp, f, indent=4)

        return scientific_texts_tmp

    # Функция для загрузки новостных текстов из News API
    def get_journalistic_texts(self):
        with open('journalistic_texts.json', 'r') as f:
            journalistic_texts_tmp = json.load(f)

        if len(journalistic_texts_tmp) > 0:
            return journalistic_texts_tmp

        urls = [
            f'https://newsapi.org/v2/everything?q=%22finland%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22greece%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22canada%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22usa%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22portugal%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22russia%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22china%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22japan%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22hungary%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22swiss%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22mexico%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
            f'https://newsapi.org/v2/everything?q=%22lithuania%22&pageSize=100&from=2024-05-20&language=en&apiKey=d2f910f7263b4091897dce5faa66b5e6'
        ]
        for url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                for article in articles:
                    journalistic_texts_tmp.append(article['content'])

        with open('journalistic_texts.json', 'w') as f:
            json.dump(journalistic_texts_tmp, f, indent=4)

        return journalistic_texts_tmp

    # Функция для загрузки официальных текстов
    def get_official_texts(self):
        with open('official_texts.json', 'r', encoding='utf-8') as f:
            official_texts_tmp = json.load(f)

        if len(official_texts_tmp) > 0:
            return official_texts_tmp

        urls = [
            'https://gutenberg.org/files/5/5.txt',     # The United States Constitution
            'https://www.gutenberg.org/files/1/1.txt'  # The United States Declaration of Independence
            'https://gutenberg.org/files/2/2.txt'      # The United States Bill of Rights
        ]
        for url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                official_texts_tmp.append(response.text)

        with open('official_texts.json', 'w') as f:
            json.dump(official_texts_tmp, f, indent=4)

        return official_texts_tmp

    # Функция для предобработки текста
    def preprocess_text(self, text):
        if isinstance(text, str):
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            return ' '.join(tokens)
        return ''

    def train(self):
        # Создание DataFrame
        df = pd.DataFrame({'text': self.texts, 'label': self.labels})

        # Предобработка текстов
        df['preprocessed_text'] = df['text'].apply(self.preprocess_text)

        # Преобразование текстов в TF-IDF матрицу
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(df['preprocessed_text'])
        y = df['label']

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Подбор гиперпараметров
        param_grid = {
            'alpha': [0.1, 0.5, 1.0]
        }

        self.model = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
        self.model.fit(X_train, y_train)

        print("Best parameters found: ", self.model.best_params_)
        print("Best cross-validation score: ", self.model.best_score_)

        # Предсказание на тестовой выборке
        y_pred = self.model.predict(X_test)

        # Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # plt.show()

        # Оценка точности и других метрик
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def classify_text_parts(self, text, main_style):
        # Разделение текста на предложения
        sentences = nltk.sent_tokenize(text)

        # Преобразование предложений в TF-IDF матрицу
        X_sentences = self.vectorizer.transform(sentences)

        # Классификация предложений
        predictions = self.model.predict(X_sentences)

        # Выявление предложений, стиль которых не соответствует основному
        return [sentences[i] for i in range(len(sentences)) if predictions[i] != main_style]

    def save(self):
        joblib.dump(self.model, 'trained_model.pkl')
        joblib.dump(self.vectorizer, 'trained_vectorizer.pkl')

    def load(self):
        self.model = joblib.load('trained_model.pkl')
        self.vectorizer = joblib.load('trained_vectorizer.pkl')


# Пример текста
new_text = """
At first I did not reply to this question, and every clean-minded man
in my place would have hesitated too. The Count was fond of me, and
quite sincerely obtruded his friendship on me. I, on my part, felt
nothing like friendship for the Count; I even disliked him. It would
therefore have been more honest to reject his friendship once for all
than to go to him and dissimulate. Besides, to go to the Count's meant
to plunge once more into the life my Polycarp had characterized as a
“pigsty,” which two years before during the Count's residence on his
estate and until he left for Petersburg had injured my good health and
had dried up my brain. That loose, unaccustomed life so full of show
and drunken madness, had not had time to shatter my constitution, but
it had made me notorious in the whole Government
"""

# # Классификация частей текста и выявление несоответствий
# text_model = TextModel()
# text_model.train()
# text_model.save()
# text_model.load()
#
# # Основной стиль текста (например, "научный")
# main_style = "official"
# inconsistent_sentences = text_model.classify_text_parts(new_text, main_style)
# print("Inconsistent sentences:", inconsistent_sentences)
#
# # Основной стиль текста (например, "научный")
# main_style = "jornalistic"
# inconsistent_sentences = text_model.classify_text_parts(new_text, main_style)
# print("Inconsistent sentences:", inconsistent_sentences)