import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

base_model = ResNet50(weights="imagenet", include_top=False)
x = GlobalAveragePooling2D()(base_model.output)
feature_model = Model(base_model.input, x)

def extract_feature(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(np.expand_dims(img, axis=0))
    feat = feature_model.predict(img)[0]
    feat = feat / np.linalg.norm(feat)
    return feat

def recommend(query_feat, myntra_feats, myntra_names, top_k=5):
    sim = cosine_similarity(query_feat.reshape(1,-1), myntra_feats)[0]
    idx = np.argsort(sim)[-top_k:][::-1]
    return [(myntra_names[i], float(sim[i])) for i in idx]