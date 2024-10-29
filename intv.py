import os
import re
import nltk
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from gensim.models import Word2Vec
from wordcloud import WordCloud

from transformers import pipeline, AutoTokenizer, BertModel, BertForSequenceClassification, TextClassificationPipeline
from sentence_transformers import SentenceTransformer

import torch
import logging
from tqdm import tqdm
import traceback
import gc
import json
from fpdf import FPDF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

device = 0 if torch.cuda.is_available() else -1
device_name = 'cuda' if device == 0 else 'cpu'
logging.info(f"Using device: {device_name}")

logging.info("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_trf") 
except OSError:
    logging.info("spaCy model not found. Downloading 'en_core_web_trf'...")
    from spacy.cli import download
    download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")

logging.info("Initializing sentiment analysis pipeline...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

logging.info("Initializing aspect-based sentiment analysis pipeline...")
aspect_sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)

logging.info("Loading BERT tokenizer and model for feature extraction...")
feature_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_feature_model = BertModel.from_pretrained("bert-base-uncased").to(device_name)

logging.info("Loading SentenceTransformer model for embeddings...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device_name)

logging.info("Loading Personality Prediction model...")
personality_tokenizer = AutoTokenizer.from_pretrained("Minej/bert-base-personality")
personality_model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality").to(device_name)

def read_file(file_path):
    """Read the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_files(path):
    """Read a single file or all .txt files in a directory."""
    if os.path.isfile(path):
        return [read_file(path)]
    elif os.path.isdir(path):
        files = []
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                file_path = os.path.join(path, filename)
                files.append(read_file(file_path))
        return files
    else:
        raise ValueError("Invalid path. Please provide a valid file or directory path.")

def preprocess_text(text):
    """Preprocess text by removing non-alphabetic characters, tokenizing, removing stopwords, and lemmatizing."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

def extract_features(text):
    """Extract features from text using BERT."""
    inputs = feature_tokenizer.encode_plus(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device_name) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_feature_model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return features

def chunk_text(text, chunk_size=256, overlap=128):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def analyze_sentiment_chunked(text):
    """Analyze overall sentiment of the text using chunking and sliding window."""
    chunks = chunk_text(text)
    sentiments = []
    for chunk in chunks:
        try:
            result = sentiment_analyzer(chunk[:512])[0]  # Limit each chunk to 512 tokens
            sentiments.append((result['label'], result['score']))
        except Exception as e:
            logging.error(f"Sentiment analysis failed for a chunk: {e}")
    
    if not sentiments:
        return "Unknown", 0.0
    
    positive_score = sum(score for label, score in sentiments if label == 'POSITIVE')
    negative_score = sum(score for label, score in sentiments if label == 'NEGATIVE')
    
    total_chunks = len(sentiments)
    if positive_score > negative_score:
        return "POSITIVE", positive_score / total_chunks
    elif negative_score > positive_score:
        return "NEGATIVE", negative_score / total_chunks
    else:
        return "NEUTRAL", 0.5


def analyze_aspect_sentiment(text, aspects):
    """Analyze sentiment for specific aspects of the text."""
    chunks = chunk_text(text, chunk_size=256, overlap=128)  # Smaller chunks
    aspect_sentiments = {aspect: [] for aspect in aspects}
    
    for chunk in chunks:
        aspect_texts = [f"{aspect}: {chunk}" for aspect in aspects]
        try:
            for aspect_text in aspect_texts:
                result = aspect_sentiment_analyzer(aspect_text[:512])[0]  # Limit to 512 tokens
                aspect = aspect_text.split(':')[0].strip()
                aspect_sentiments[aspect].append((result['label'], result['score']))
        except Exception as e:
            logging.error(f"Error analyzing sentiment for aspects in a chunk: {e}")
    
    aggregated_sentiments = {}
    for aspect, sentiments in aspect_sentiments.items():
        if not sentiments:
            aggregated_sentiments[aspect] = {"label": "Unknown", "score": 0.0}
        else:
            avg_score = sum(float(score) for _, score in sentiments) / len(sentiments)
            label = max(set(label for label, _ in sentiments), key=lambda x: sum(1 for l, _ in sentiments if l == x))
            aggregated_sentiments[aspect] = {"label": label, "score": avg_score}
    
    return aggregated_sentiments

def extract_named_entities(text):
    """Extract named entities from text using spaCy's transformer-based NER."""
    chunks = chunk_text(text)
    all_entities = []
    for chunk in chunks:
        doc = nlp(chunk)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        all_entities.extend(entities)
    
    entity_counts = {}
    for entity in all_entities:
        key = (entity['text'], entity['label'])
        entity_counts[key] = entity_counts.get(key, 0) + 1
    
    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return [{"text": k[0], "label": k[1], "count": v} for (k, v) in top_entities]

def personality_detection_chunked(text):
    """Detect personality traits from text using chunking and sliding window."""
    chunks = chunk_text(text)
    trait_scores = {
        "Extroversion": [],
        "Neuroticism": [],
        "Agreeableness": [],
        "Conscientiousness": [],
        "Openness": []
    }
    
    for chunk in chunks:
        if len(chunk.strip()) == 0:
            continue
        
        inputs = personality_tokenizer.encode_plus(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        inputs = {k: v.to(device_name) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = personality_model(**inputs)
        
        logits = outputs.logits.squeeze()
        probabilities = torch.sigmoid(logits).cpu().tolist()
        
        for trait, score in zip(trait_scores.keys(), probabilities):
            trait_scores[trait].append(score)
    
    aggregated_traits = {trait: sum(scores) / len(scores) if scores else 0.0 
                         for trait, scores in trait_scores.items()}
    
    total_score = sum(aggregated_traits.values())
    normalized_traits = {trait: score / total_score for trait, score in aggregated_traits.items()}
    
    return normalized_traits

def extract_personality_traits(text):
    """Extract personality traits from text."""
    return personality_detection_chunked(text)
def extract_personality_traits(text):
    """Extract personality traits from text."""
    return personality_detection_chunked(text)

def generate_word_embeddings(texts):
    """Generate word embeddings using Word2Vec."""
    preprocessed_texts = [preprocess_text(text) for text in texts]
    model = Word2Vec(sentences=preprocessed_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model

def perform_topic_modeling(texts, n_topics=10):
    """Perform topic modeling using Latent Dirichlet Allocation."""
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1,2),
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42, 
        learning_method='batch', 
        max_iter=50
    )
    lda_output = lda_model.fit_transform(tfidf_matrix)
    return lda_model, lda_output, vectorizer

def plot_sentiment(sentiment, filename='sentiment_distribution.png'):
    """Plot sentiment distribution and save as an image."""
    labels = ['Negative', 'Neutral', 'Positive']
    scores = [0, 0, 0]
    label = sentiment['label'].upper()
    score = sentiment['score']
    if label == 'POSITIVE':
        scores[2] = score
        scores[1] = 1 - score
    elif label == 'NEGATIVE':
        scores[0] = score
        scores[1] = 1 - score
    else:
        scores[1] = score
    sns.barplot(x=labels, y=scores)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_personality_traits(traits, filename='personality_traits.png'):
    """Plot personality traits and save as an image."""
    traits_names = list(traits.keys())
    traits_scores = list(traits.values())
    sns.barplot(x=traits_names, y=traits_scores)
    plt.title('Personality Traits')
    plt.xlabel('Traits')
    plt.ylabel('Normalized Scores')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# def generate_wordcloud(text, filename='wordcloud.png'):
#     """Generate a word cloud from text and save as an image."""
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(15, 7.5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()

def generate_report(results):
    """Generate a PDF report consolidating all analysis results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add Topics
    # pdf.cell(200, 10, txt="Topics", ln=True, align='C')
    # for idx, topic in enumerate(results['topics']):
    #     pdf.multi_cell(0, 10, f"Topic {idx+1}: {', '.join(topic)}")

    pdf.cell(200, 10, txt="Overall Sentiment", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Label: {results['sentiment']['label']}", ln=True)
    pdf.cell(200, 10, txt=f"Score: {results['sentiment']['score']:.2f}", ln=True)

    pdf.cell(200, 10, txt="Aspect-Based Sentiment", ln=True, align='C')
    for aspect, sentiment in results['aspect_sentiment'].items():
        pdf.cell(200, 10, txt=f"{aspect}: {sentiment['label']} (Score: {sentiment['score']:.2f})", ln=True)

    pdf.cell(200, 10, txt="Personality Traits", ln=True, align='C')
    for trait, score in results['personality_traits'].items():
        pdf.cell(200, 10, txt=f"{trait}: {score:.3f}", ln=True)

    # # Add Named Entities
    # pdf.cell(200, 10, txt="Named Entities", ln=True, align='C')
    # if results['named_entities']:
    #     for entity in results['named_entities']:
    #         pdf.cell(200, 10, txt=f"{entity['text']} ({entity['label']}) - Count: {entity['count']}", ln=True)
    # else:
    #     pdf.cell(200, 10, txt="No named entities found.", ln=True)

    # pdf.add_page()
    # pdf.cell(200, 10, txt="Visualizations", ln=True, align='C')
    # pdf.image('sentiment_distribution.png', x=10, y=20, w=190)
    # pdf.add_page()
    # pdf.image('personality_traits.png', x=10, y=20, w=190)
    # pdf.add_page()
    # pdf.image('wordcloud.png', x=10, y=20, w=190)

    pdf.output("interview_analysis_report.pdf")
    logging.info("Report generated: 'interview_analysis_report.pdf'")

def analyze_interviews(interviews):
    """Analyze a single interview."""
    if len(interviews) != 1:
        raise ValueError("This analysis is designed for a single interview.")

    interview = interviews[0]

    try:

        logging.info("Preprocessing interview text...")
        preprocessed_interview = preprocess_text(interview)

        logging.info("Extracting BERT features...")
        bert_features = extract_features(interview)
        gc.collect()

        logging.info("Generating word embeddings...")
        word2vec_model = generate_word_embeddings([interview])
        gc.collect()

        logging.info("Performing topic modeling...")
        lda_model, lda_output, vectorizer = perform_topic_modeling([interview], n_topics=10)
        topics = []
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            logging.info(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
            topics.append(top_words)
        gc.collect()

        logging.info("Generating word cloud...")

        gc.collect()

        logging.info("Performing Named Entity Recognition...")
        entities = extract_named_entities(interview)
        if entities:
            logging.info(f"Top Entities: {entities}")
        else:
            logging.info("No named entities found in the interview.")
        gc.collect()

        logging.info("Performing Overall Sentiment Analysis...")
        label, score = analyze_sentiment_chunked(interview)
        logging.info(f"Overall sentiment: {label} (score: {score:.2f})")

        logging.info("Performing Aspect-Based Sentiment Analysis...")
        aspects = ["Career Satisfaction", "Challenges", "Personal Growth"]
        aspect_sentiment = analyze_aspect_sentiment(interview, aspects)
        for aspect, sentiment in aspect_sentiment.items():
            logging.info(f"{aspect} Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")

        logging.info("Extracting Personality Traits...")
        personality_traits = extract_personality_traits(interview)
        logging.info(f"Personality Traits: {personality_traits}")

        logging.info("Identifying most similar words based on Word2Vec embeddings...")
        for word in ['research', 'career', 'academia', 'industry']:
            try:
                similar_words = word2vec_model.wv.most_similar(word, topn=5)
                logging.info(f"Words similar to '{word}': {similar_words}")
            except KeyError:
                logging.info(f"'{word}' not found in the vocabulary.")
        gc.collect()

        logging.info("Generating visualizations...")
        # plot_sentiment({'label': label, 'score': score})
        # plot_personality_traits(personality_traits)

        results = {
            'topics': topics,
            'named_entities': entities,
            'sentiment': {'label': label, 'score': score},
            'aspect_sentiment': aspect_sentiment,
            'personality_traits': personality_traits
        }

        logging.info("Generating PDF report...")
        generate_report(results)

        with open('interview_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        logging.info("Analysis results saved to 'interview_analysis_results.json'")

    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")
        logging.error(traceback.format_exc())

def main(interview_path):
    """Main function to read interview data and perform analysis."""
    try:
        logging.info(f"Reading interview data from: {interview_path}")
        interviews = read_files(interview_path)
        analyze_interviews(interviews)
    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python intv.py <interview_file_or_directory>")
        sys.exit(1)
    interview_path = sys.argv[1]
    main(interview_path)