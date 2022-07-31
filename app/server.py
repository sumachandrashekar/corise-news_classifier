import datetime
import json

from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib
import numpy as np
import time

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "../data/news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "../data/logs.out"
    }
}

class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, sentence_transformer_model):
        self.sentence_transformer_model = sentence_transformer_model

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, featurizer_path: str, classfier_path: str) -> None:
        sentence_transformer_model = SentenceTransformer(
            'sentence-transformers/{model}'.format(model=featurizer_path)
        )
        featurizer = TransformerFeaturizer(sentence_transformer_model=sentence_transformer_model)
        classifier = joblib.load(classfier_path)

        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', classifier)
        ])

    def predict_proba(self, model_input: str) -> dict:
        prediction = self.pipeline.predict_proba(np.array([model_input]))

        return {l: p for l, p in zip(self.pipeline.classes_, prediction[0])}

    def predict_label(self, model_input: dict) -> str:
        return self.pipeline.predict(np.array([model_input]))[0]


app = FastAPI()

classifier_pipeline = None
log_file = None

@app.on_event("startup")
def startup_event():
    global classifier_pipeline
    classifier_pipeline = NewsCategoryClassifier(
        featurizer_path=GLOBAL_CONFIG['model']['featurizer']['sentence_transformer_model'],
        classfier_path=GLOBAL_CONFIG['model']['classifier']['serialized_model_path']
    )

    global log_file
    log_file = open(GLOBAL_CONFIG['service']['log_destination'], 'a')

    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    log_file.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()

    probabilities = classifier_pipeline.predict_proba(request.description)
    label = classifier_pipeline.predict_label(request.description)

    latency = time.time() - start_time

    response = PredictResponse(
        scores=probabilities,
        label=label
    )

    timestamp = datetime.datetime.now()

    log_file.write(
        '\n'+json.dumps(
            {
                'timestamp': timestamp.strftime("%Y:%M:%D, %H:%M:%S"),
                'request': request.json(),
                'prediction': response.json(),
                'latency': latency
            }
        )
    )

    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
