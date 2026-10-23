from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    return Pipeline([('tfidf',TfidfVectorizer(max_features=50000,ngram_range=(1,2),min_df=2,sublinear_tf=True)),('classifier',LogisticRegression(max_iter=1000,random_state=42))])
#Feature engineering script
