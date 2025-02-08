from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from fastapi.middleware.cors import CORSMiddleware

# Télécharger les stopwords en arabe
#nltk.download('stopwords')
#nltk.download('punkt')

nltk_data_dir = "C:/Users/Test/OneDrive/Bureau/ING3/AND/projects/p2/.venv/nltk_data"
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)


app = FastAPI()
# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines (à adapter en production)
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

# Liste des stopwords en arabe
arabic_stopwords = stopwords.words('arabic')

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(r"[ًٌَُِّْ~]", "", text)
    text = re.sub(r"ـ", "", text)
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^ء-ي\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def remove_stopwords(text, stopwords):
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stopwords])

def preprocess_text(text, stopword_list):
    text = remove_punctuations(text)
    text = normalize_arabic(text)
    text = clean_text(text)
    text = remove_repeating_char(text)
    text = remove_stopwords(text, stopword_list)
    return text

def preprocess_corpus(dataframe, column_name, stopword_list):
    dataframe[column_name] = dataframe[column_name].apply(lambda x: preprocess_text(x, stopword_list))
    return dataframe

# Modèle et vectorizer globaux
tfidf_vectorizer = None
mlp = None
pca = None

@app.post("/preprocess-and-train")
async def preprocess_and_train():
    global tfidf_vectorizer, mlp ,pca
    try:
        file_path = "C:/Users/Test/OneDrive/Bureau/ING3/AND/projects/p2/src/AJGT.xlsx"
        df = pd.read_excel(file_path, engine="openpyxl")
        #df = pd.read_excel("C:\\Users\Test\OneDrive\Bureau\ING3\AND\projects\p2\src\AJGT.xlsx")
        df_copy = df.copy()
        df_copy = preprocess_corpus(df_copy, "Feed", arabic_stopwords)

        tokenizer = RegexpTokenizer(r'\w+')
        df_copy["Feed"] = df_copy["Feed"].apply(tokenizer.tokenize)

        texts = df_copy['Feed']
        texts = texts.apply(lambda x: " ".join(x) if isinstance(x, list) else x)

        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b"
        )

        X_tfidf = tfidf_vectorizer.fit_transform(texts)
        X_tfidf_array = X_tfidf.toarray()

        pca = PCA(n_components=1400)
        X_pca = pca.fit_transform(X_tfidf_array)

        # Supposons que 'y' est une colonne dans votre dataframe
        y = df_copy['Sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

        mlp = MLPClassifier(hidden_layer_sizes=(1000, 50), max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)

        train_predictions = mlp.predict(X_train)
        test_predictions = mlp.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        classification_report_result = classification_report(y_test, test_predictions)

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": classification_report_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Modèle Pydantic pour la requête de classification
class Comment(BaseModel):
    text: str

@app.post("/classify-comment")
async def classify_comment(comment: Comment):
    global tfidf_vectorizer, mlp, pca  # Ajoutez pca aux variables globales
    if tfidf_vectorizer is None or mlp is None or pca is None:
        raise HTTPException(status_code=400, detail="Le modèle n'est pas encore entraîné.")

    try:
        # Prétraitement du commentaire
        processed_text = preprocess_text(comment.text, arabic_stopwords)
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized_text = tokenizer.tokenize(processed_text)
        text_final = " ".join(tokenized_text)

        # Transformation TF-IDF
        X_tfidf = tfidf_vectorizer.transform([text_final])
        X_tfidf_array = X_tfidf.toarray()


        # Réduction de dimension avec PCA (appliquer le PCA entraîné)
        X_pca = pca.transform(X_tfidf_array)

        # Prédiction
        prediction = mlp.predict(X_pca)
        print("Prédiction :", prediction[0])
        sentiment = "positif" if prediction[0] == "Positive" else "négatif"

        return {"comment": comment.text, "sentiment": sentiment}
    except Exception as e:
        print(f"Erreur lors de la classification : {e}")
        raise HTTPException(status_code=500, detail=str(e))
# Interface HTML pour saisir un commentaire
@app.get("/", response_class=HTMLResponse)
async def get_interface():
    return """
    <html>
        <head>
            <title>Classification de Commentaires</title>
        </head>
        <body>
            <h1>Classer un commentaire</h1>
            <form id="commentForm">
                <label for="comment">Entrez votre commentaire :</label><br>
                <textarea id="comment" name="comment" rows="4" cols="50"></textarea><br><br>
                <button type="button" onclick="classifyComment()">Classer</button>
            </form>
            <h2>Résultat :</h2>
            <p id="result"></p>
            <script>
                async function classifyComment() {
                    const comment = document.getElementById("comment").value;
                    const response = await fetch("/classify-comment", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({ text: comment }),
                    });
                    const data = await response.json();
                    document.getElementById("result").innerText = "Sentiment : " + data.sentiment;
                }
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)