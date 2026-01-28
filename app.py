"""
Content Clustering Web Application
==================================
A Flask web app that clusters content from Screaming Frog exports
using OpenAI embeddings and k-means clustering.
"""

import os
import io
import json
import time
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
RESULTS_FOLDER = '/tmp/results'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_columns(df):
    """Auto-detect common Screaming Frog column names."""
    column_mapping = {
        'url': None,
        'title': None,
        'h1': None,
        'meta_description': None
    }
    
    # Common variations
    url_patterns = ['address', 'url', 'page url', 'page']
    title_patterns = ['title 1', 'title', 'page title', 'meta title']
    h1_patterns = ['h1-1', 'h1', 'heading 1', 'h1 1']
    meta_patterns = ['meta description 1', 'meta description', 'description']
    
    columns_lower = {col.lower(): col for col in df.columns}
    
    for pattern in url_patterns:
        if pattern in columns_lower:
            column_mapping['url'] = columns_lower[pattern]
            break
    
    for pattern in title_patterns:
        if pattern in columns_lower:
            column_mapping['title'] = columns_lower[pattern]
            break
    
    for pattern in h1_patterns:
        if pattern in columns_lower:
            column_mapping['h1'] = columns_lower[pattern]
            break
    
    for pattern in meta_patterns:
        if pattern in columns_lower:
            column_mapping['meta_description'] = columns_lower[pattern]
            break
    
    return column_mapping


def prepare_text_for_embedding(df, column_mapping):
    """Combine text columns for embedding."""
    text_parts = []
    
    for key in ['url', 'title', 'h1', 'meta_description']:
        col = column_mapping.get(key)
        if col and col in df.columns:
            text_parts.append(df[col].fillna('').astype(str))
    
    if not text_parts:
        raise ValueError("No valid text columns found for clustering")
    
    combined = text_parts[0]
    for part in text_parts[1:]:
        combined = combined + ' ' + part
    
    return combined.str.strip()


def get_embeddings(texts, api_key):
    """Get OpenAI embeddings for texts."""
    client = OpenAI(api_key=api_key)
    embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        
        time.sleep(0.3)  # Rate limiting
    
    return np.array(embeddings)


def find_optimal_clusters(embeddings, min_k=3, max_k=15):
    """Find optimal cluster count using silhouette score."""
    best_score = -1
    best_k = min_k
    
    max_k = min(max_k, len(embeddings) - 1)
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k


def generate_cluster_names(df, cluster_col, title_col, url_col, api_key):
    """Use GPT to generate cluster names."""
    client = OpenAI(api_key=api_key)
    cluster_names = {}
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_id].head(10)
        
        samples = []
        for _, row in cluster_df.iterrows():
            title = row.get(title_col, '') if title_col else ''
            url = row.get(url_col, '') if url_col else ''
            samples.append(f"- {title} ({url})")
        
        sample_text = "\n".join(samples)
        
        prompt = f"""Based on these page samples, suggest a short descriptive name (2-4 words) for this content group.
Return ONLY the cluster name.

Pages:
{sample_text}

Cluster name:"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3
            )
            cluster_names[cluster_id] = response.choices[0].message.content.strip()
        except:
            cluster_names[cluster_id] = f"Cluster {cluster_id + 1}"
    
    return cluster_names


def create_visualization_data(df, embeddings):
    """Create 2D coordinates for visualization."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    return coords[:, 0].tolist(), coords[:, 1].tolist()


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload and return column detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(filepath)
        
        # Read and analyze CSV
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
