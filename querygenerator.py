import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
nltk.download('words')
from nltk.corpus import words
import spacy
from spacy.matcher import PhraseMatcher, Matcher

def preprocess(naturallang):
    naturallang=naturallang.replace("marks","")
    naturallang=naturallang.replace("are","")
    naturallang=naturallang.replace("to","")
    pattern = r'(\d+)(st|nd|rd|th)'
    naturallang=re.sub(pattern, r'\1', naturallang)
    naturallang=naturallang.replace("grade","class")
    words = nltk.word_tokenize(naturallang)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    # words = word_tokenize(lemmatized_text)
    # stop_words = set(stopwords.words('english'))
    # stop_words.remove("a")
    # filtered_words = [word for word in words if word.lower() not in stop_words]
    # gg=' '.join(filtered_words)
    gg=lemmatized_text
    gg=gg.replace('id','ID')
    # g=gg.replace("'","").replace(",","").replace("  "," ")
    return gg
#need this!!

english_words = set(words.words())
english_words.remove("a")
def is_english_word(word):
    return word.lower() in english_words

df=pd.read_csv("studentdatset.csv")
queries=df["Query"]
intents=df["Intent"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents)
X_train, X_test, y_train, y_test = train_test_split(queries, y, test_size=0.2, random_state=42)

# Create and train model
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)

# Predict intent
def predict_intent(query):
    intent_idx = model.predict([query])[0]
    return label_encoder.inverse_transform([intent_idx])[0]


def getentities(text):
# Load a spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Initialize the PhraseMatcher for columns
    phrase_matcher = PhraseMatcher(nlp.vocab)
    columns = ["name", "id", "age", "grade", "section", "english", "hindi", "science"]
    patterns = [nlp(text) for text in columns]
    phrase_matcher.add("COLUMN", None, *patterns)

    # Initialize the Matcher for conditions
    matcher = Matcher(nlp.vocab)
    condition_patterns = [
        [{"LOWER":{"IN":columns}},{"LOWER": {"IN": ["greater", "less", "equal"]}}, {"LOWER": "than"}, {"IS_DIGIT": True}],  # e.g., "greater than 18"
        [{"LOWER":{"IN":columns}},{"LOWER": "equal"}, {"LOWER": "to"}, {"IS_DIGIT": True}],
        [{"LOWER":{"IN":columns}},{"LOWER": "equal"}, {"LOWER": "to"}, {"IS_ALPHA": True}],
        [{"LOWER": {"IN": ["older", "younger", "equal"]}}, {"LOWER": "than"}, {"IS_DIGIT": True}],
        [{"LOWER":'id'},{"IS_DIGIT": True}]
    ]
    dbcolumns=["age","name","section","class","id","english","hindi","science"]
    patterns=[]
    for colm in dbcolumns:
        patterns.append([{"LOWER": colm}, {"IS_ALPHA":True}])
        patterns.append([{"LOWER": colm}, {"IS_DIGIT":True}])
    condition_patterns+=patterns


    matcher.add("CONDITION", condition_patterns)

    # Process a text
    # text="Get names of students whose english marks are greater than 40"

    pretext=preprocess(text)

    doc = nlp(pretext)
    # Apply the PhraseMatcher and Matcher to the doc
    column_matches = phrase_matcher(doc)
    condition_matches = matcher(doc)
    finalcolumns=[]
    # Print the column matches
    for match_id, start, end in column_matches:
        span = doc[start:end]
        finalcolumns.append(span.text)

    finalconditions=[]
    # Print the condition matches
    for match_id, start, end in condition_matches:
        span = doc[start:end]
        word=span.text.split(" ")
        if len(word)==2: 
            if is_english_word(word[1]):
                pass
            else:
                # print(word[1])
                if not word[1].isdigit:
                    word[1]="\'"+word[1]+"\'"
                finalconditions.append("=".join(word))
        else:
            finalconditions.append(span.text)
    for i in finalcolumns:
        for j in finalconditions:
            if i in j:
                finalcolumns.remove(i)
    relations={
        "greater than":">",
        "less than":"<",
        "equal to":"=",
        "is equal to":"=",
        "older than":"age>",
        "younger than":"age<"
    }
    for i in range(0,len(finalconditions)):
        for key, value in relations.items():
            finalconditions[i]=finalconditions[i].replace(key,value)
    return finalcolumns, finalconditions
# print(finalcolumns)
# print(finalconditions)

def genquery(finalconditions,finalcolumns,intent):
    query=""
    if intent=="SELECT":
        cols=""
        condi=""
        for i in range(0,len(finalcolumns)-1):
            cols+=i+","
        cols+=finalcolumns[len(finalcolumns)-1]
        for i in range(0,len(finalconditions)-1):
            condi+=i+" and"
        condi+=finalconditions[len(finalconditions)-1]

        
        query="SELECT "+cols+" FROM students"
        if condi!="":
            query+=" WHERE "+condi+";"
    elif intent=="UPDATE":
        colnames=""
        val=""
        for i in range(len(finalconditions)-1):
            val+=finalconditions[i]
        colnames+=finalconditions[len(finalconditions)-1]
        query="UPDATE students SET "+ val+ " WHERE "+ colnames+";"
    elif intent=="DELETE":
        gg=""
        for i in finalconditions:
            i=i.split("=")
            if i[1].isdigit():
                gg+=i[0]+"="+i[1]
            else:
                gg+=i[0]+"="+"\'"+i[1]+"\'"
        query="DELETE FROM students WHERE "+gg+";"

    elif intent=="INSERT":
        colnames=""
        val=""
        for u in range(0,len(finalconditions)):
            if u==len(finalconditions)-1:
                g=finalconditions[u].split("=")
                colnames+=g[0]
                if g[1].isdigit():
                    val+=g[1]+","
                else:
                    val+="\'"+g[1]+"\'"+","
            else:
                g=finalconditions[u].split("=")
                colnames+=g[0]+","
                if g[1].isdigit():
                    val+=g[1]+","
                else:
                    val+="\'"+g[1]+"\'"+","
        query="INSERT INTO students ("+colnames+") VALUES ("+val+");"
    query=query.replace('ID','id')  
    return query
# print(query)

def printquery(text):
    print("given input: ",end="")
    print(preprocess(text))
    intent=predict_intent(text)
    print("Intent: ",end="")
    print(intent)
    finalcolumns,finalconditions=getentities(text)
    print("columns found: ",end="")
    print(finalcolumns)
    print("conditions found: ",end="")
    print(finalconditions)
    print("Final query: ",end="")
    print(genquery(finalconditions,finalcolumns,intent))


print("Warning: Dont enter special charectericstics!")
text=input("enter query in natural language")
printquery(text)
