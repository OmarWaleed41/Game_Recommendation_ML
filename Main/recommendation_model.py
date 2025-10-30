import sys
import os
# Must be BEFORE any other imports!
if sys.platform == 'win32':
    torch_lib = os.path.join(os.path.dirname(sys.executable), 
                             'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
    import torch

import pickle
import csv
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


# we in the future want to cache the entire model and not recompute it every time using and we can use CSV hash checking to see if we need to recompute it
# well we did that part but now we need to see when do we need to recompute it if let's say a new entry is added to the dataset or some values got updated

# Smart preprocessing - separate core gameplay tags from artistic tags
def preprocess_field_smart(text, weight=1, boost_core=True):
    """Smart preprocessing that identifies core gameplay tags"""
    if not text or pd.isna(text):
        return ""
    
    # Core gameplay tags to boost their weight
    core_tags = {
        'souls_like', 'souls-like', 'soulslike', 'metroidvania', 'roguelike', 'roguelite',
        'platformer', 'action_rpg', 'rpg', 'hack_and_slash', 'turn-based', 'turn_based',
        'difficult', 'precision_platformer', 'side_scroller', 'top-down', 'top_down',
        'bullet_hell', 'exploration', 'combat', 'boss_rush', 'parry', 'dodge',
        'challenging', '2d_platformer', '3d_platformer', 'fast-paced', 'fast_paced',
        'stealth', 'tactical', 'strategy', 'puzzle', 'horror', 'survival'
    }
    
    terms = [t.strip().replace(" ", "_").lower() for t in str(text).split(",") if t.strip()]
    weighted_terms = []
    
    for term in terms:
        # Check if this is a core gameplay tag
        is_core = any(core in term for core in core_tags)
        
        if is_core and boost_core:
            # Core gameplay tags get extra weight
            weighted_terms.extend([term] * (weight * 2))
        else:
            # Artistic/thematic tags get normal weight
            weighted_terms.extend([term] * weight)
    
    return " ".join(weighted_terms)

def preprocess_field(text, weight=1):
    """Standard preprocessing"""
    if not text or pd.isna(text):
        return ""
    terms = [t.strip().replace(" ", "_") for t in str(text).split(",") if t.strip()]
    weighted_terms = []
    for term in terms:
        weighted_terms.extend([term] * weight)
    return " ".join(weighted_terms)

CACHE_FILE = "recommendation_model_cache.pkl"

if os.path.exists(CACHE_FILE):
    print("Loading cached model...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
        df_filtered = cache['df_filtered']
        tfidf_matrix = cache['tfidf_matrix']
        embedding_matrix = cache['embedding_matrix']
        hybrid_matrix = cache['hybrid_matrix']
        knn = cache['knn']
    print("Model loaded from cache!")
else:
    csv.field_size_limit(2**31 - 1)

    print("Loading data...")
    sys.stdout.flush()

    columns_needed = ['Name', 'Supported languages', 'Negative', 'Score rank', 
                    'Screenshots', 'Tags', 'Genres', 'Publishers', 'Categories', 'Website']

    df = pd.read_csv("games.csv", encoding='utf-8', low_memory=False, 
                    usecols=columns_needed)

    print("\n Available columns:")
    print(df.columns.tolist())
    print()

    # The datset is misaligned, so we remap columns
    df["about_the_game"] = df['Supported languages']
    df["actual_positive"] = df['Negative']
    df["actual_negative"] = df['Score rank']
    df['detailed_tags'] = df['Screenshots']
    df['simple_genres'] = df['Tags']
    df['steam_features'] = df['Genres']
    df['developers'] = df['Publishers']
    df["publishers"] = df['Categories']
    df['game_image'] = df['Website']

    # Calculate reviews
    df['actual_positive'] = pd.to_numeric(df['actual_positive'], errors='coerce').fillna(0).astype(int)
    df['actual_negative'] = pd.to_numeric(df['actual_negative'], errors='coerce').fillna(0).astype(int)
    df['total_reviews'] = df['actual_positive'] + df['actual_negative']
    df['positive_ratio'] = df['actual_positive'] / (df['total_reviews'] + 1)

    # Filter
    MIN_REVIEWS = 800
    df_filtered = df[df['total_reviews'] >= MIN_REVIEWS].copy()

    # Fill missing values
    df_filtered['detailed_tags'] = df_filtered['detailed_tags'].fillna("")
    df_filtered['steam_features'] = df_filtered['steam_features'].fillna("")
    df_filtered['developers'] = df_filtered['developers'].fillna("")
    df_filtered['about_the_game'] = df_filtered['about_the_game'].fillna("")

    # Process features with smart weighting
    print("Processing features...")
    df_filtered['tags_processed'] = df_filtered['detailed_tags'].apply(
        lambda x: preprocess_field_smart(x, weight=5, boost_core=True)
    )
    df_filtered['features_processed'] = df_filtered['steam_features'].apply(
        lambda x: preprocess_field(x, weight=4)
    )
    df_filtered['about_processed'] = df_filtered['about_the_game'].apply(
        lambda x: preprocess_field(x, weight=3)
    )

    # Combine for TF-IDF
    df_filtered['combined_tfidf'] = (
        df_filtered['tags_processed'] + " " + 
        df_filtered['features_processed'] + " " +
        df_filtered['about_processed']
    )

    # Create natural language text for embeddings (no artificial repetition)(we added that in the smart preprocessor)
    df_filtered['combined_embedding'] = (
        df_filtered['detailed_tags'].fillna("") + ". " +
        df_filtered['steam_features'].fillna("") + ". " +
        df_filtered['about_the_game'].fillna("")
    )

    print(f"Features processed!, count: {len(df_filtered)}\n")

    # TF-IDF (for exact tag/keyword matching)
    # what it is basically is you take each document (each game entry in our case)
    # and tokenize the tags and genres and parts of the description and then see if a small amount of games mentions these tokens 
    # like metrodvania or rougelite these are high IDF terms however "action" is a low idf term cause it appearead in most of the documents

    print("\n Building TF-IDF vectors...")
    tfidf = TfidfVectorizer(
        stop_words='english', 
        max_features=30000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(df_filtered['combined_tfidf'])
    print(f"    TF-IDF shape: {tfidf_matrix.shape}")

    # Sentence Embeddings (for semantic understanding)
    # we use sentence transformers to get semantic embeddings of the game descriptions and tags
    # this allows us to capture the meaning behind the words used in descriptions and tags
    print("\n Building semantic embeddings...")
    print("   Computing embeddings (this may take a few minutes)...")

    # Use a lightweight but effective model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings in batches (it's basically a transformer model so it needs to be batched)
    batch_size = 32
    embeddings_list = []
    
    texts = df_filtered['combined_embedding'].tolist()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)
        
        if (i // batch_size) % 10 == 0:
            print(f"      Processed {i}/{len(texts)} games...")
    
    embedding_matrix = np.vstack(embeddings_list)

    # Hybrid Similarity Matrix

    print("\n Building hybrid KNN model...")

    # Normalize both matrices for fair combination

    tfidf_normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    embedding_normalized = normalize(embedding_matrix, norm='l2', axis=1)

    # combine: 50% TF-IDF (exact matches) + 50% embeddings (semantic similarity)
    # i see this is the perfect balance for our use case since we are now almost identical to the Steam recommendation engine 

    TFIDF_WEIGHT = 0.5
    EMBEDDING_WEIGHT = 0.5

    # convert to dense for combination (or use sparse operations if memory is an issue)(thank god we have enough RAM)
    print("   Combining TF-IDF and embeddings...")
    hybrid_matrix = np.hstack([
        tfidf_normalized.toarray() * TFIDF_WEIGHT,
        embedding_normalized * EMBEDDING_WEIGHT
    ])

    print(f"    Hybrid matrix shape: {hybrid_matrix.shape}")

    # Build KNN model on hybrid features
    # We chose neighbors=45 based on the elbow method k means using a plot test done in another script
    K = 45
    knn = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute')
    knn.fit(hybrid_matrix)

    print(" Model ready!\n")
    
    # Save everything
    print("Saving model cache...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({
            'df_filtered': df_filtered,
            'tfidf_matrix': tfidf_matrix,
            'embedding_matrix': embedding_matrix,
            'hybrid_matrix': hybrid_matrix,
            'knn': knn
        }, f)

# Check if a game exists

# game_check = df[df['Name'].str.contains('Cities', case=False, na=False)] # change the name inside the contains method
# print("\n Game/s in dataset:")
# print(game_check[['Name', 'actual_positive', 'actual_negative', 'total_reviews']])

# RECOMMENDATION FUNCTION
def recommend_games(library, top_k=30, diversity_penalty=0.0, quality_boost=0.5, popularity_threshold_boost=True):
    """Hybrid KNN recommendation system using TF-IDF + embeddings"""
    library_lower = [g.lower().strip() for g in library]
    
    idxs = []
    found_games = []

    print(" Matching your games:")
    for game in library_lower:
        best_match = None
        best_score = 0
        
        for idx, row in df_filtered.iterrows():
            name = row['Name'].lower()
            game_words = set(game.split())
            name_words = set(name.split())
            overlap = len(game_words & name_words)
            
            if overlap > best_score:
                best_score = overlap
                best_match = (idx, row['Name'])
        
        if best_match and best_score >= 1:
            idx, name = best_match
            idx_filtered = df_filtered.index.get_loc(idx)
            idxs.append(idx_filtered)
            found_games.append(name)
            print(f"    {name}")
        else:
            print(f"    '{game}' not found")

    if not idxs:
        print("\n No games found!\n")
        return pd.DataFrame()

    # Collect KNN neighbors for each game using hybrid similarity
    scores = {}
    for lib_idx in idxs:
        distances, neighbors = knn.kneighbors([hybrid_matrix[lib_idx]])
        for dist, n_idx in zip(distances[0], neighbors[0]):
            if n_idx not in idxs:
                sim = 1 - dist
                scores[n_idx] = scores.get(n_idx, 0) + sim

    if not scores:
        print("\n No neighbor results found!\n")
        return pd.DataFrame()
    
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [i for i, _ in sorted_candidates][:min(len(sorted_candidates), top_k * 4)]

    # Multi-factor scoring
    review_quality = df_filtered['positive_ratio'].values
    raw_popularity = df_filtered['actual_positive'].values
    total_reviews = df_filtered['total_reviews'].values

    popularity = np.log1p(raw_popularity)
    popularity_normalized = popularity / (popularity.max() + 1e-10)

    quality_score = review_quality * 0.6 + popularity_normalized * 0.4

    # Apply quality boost to push some highly rated games up
    for i in range(len(candidates)):
        base_sim = scores[candidates[i]]
        q = quality_score[candidates[i]]
        
        quality_multiplier = 1 + (quality_boost * (q ** 1.5))
        
        if popularity_threshold_boost:
            reviews = total_reviews[candidates[i]]
            pos_ratio = review_quality[candidates[i]]
            
            if reviews >= 100000 and pos_ratio >= 0.90:
                quality_multiplier *= 1.8
            elif reviews >= 50000 and pos_ratio >= 0.90:
                quality_multiplier *= 1.6
            elif reviews >= 20000 and pos_ratio >= 0.90:
                quality_multiplier *= 1.4
            elif reviews >= 10000 and pos_ratio >= 0.90:
                quality_multiplier *= 1.2
        
        scores[candidates[i]] = base_sim * quality_multiplier

    candidates = sorted(candidates, key=lambda x: scores[x], reverse=True)

    # Diversity filtering (we might add a condition to see how many unique developers/genres are in the library to adjust these values)
    dev_counts = {}
    genre_counts = {}
    selected = []
    
    max_per_dev = max(8, int(top_k * 0.35)) if diversity_penalty > 0 else 999
    max_per_genre = int(top_k * 0.4) if diversity_penalty > 0 else 999

    for idx in candidates:
        if len(selected) >= top_k:
            break
        
        dev = str(df_filtered.iloc[idx]['developers']).split(',')[0].strip()
        if diversity_penalty > 0 and dev:
            if dev_counts.get(dev, 0) >= max_per_dev:
                continue
            dev_counts[dev] = dev_counts.get(dev, 0) + 1
        
        genres = str(df_filtered.iloc[idx]['simple_genres']).split(',')
        main_genre = genres[0].strip() if genres else "Other"
        if diversity_penalty > 0:
            if genre_counts.get(main_genre, 0) >= max_per_genre:
                continue
            genre_counts[main_genre] = genre_counts.get(main_genre, 0) + 1
        
        selected.append(idx)

    result = df_filtered.iloc[selected][[
        'Name', 'total_reviews', 'actual_positive', 
        'actual_negative', 'positive_ratio', 'game_image'
    ]].copy()

    result['score'] = [scores[i] for i in selected]
    result = result.sort_values('score', ascending=False)

    return result

# RUN RECOMMENDATIONS

if __name__ == "__main__":
    import json
    
    # Read JSON input from stdin (the data coming from the api request)
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            user_library = data.get('games', ["Hollow Knight"])
        else:
            user_library = ["Hollow Knight"]  # Default (better than empty and also we can just throw an error)
    except:
        user_library = ["Hollow Knight"]  # Default on error (again like the condition above)
    
    recs = recommend_games(
        user_library, 
        top_k=80, 
        diversity_penalty=0.0,
        quality_boost=0.5,
        popularity_threshold_boost=True
    )
    
    # Quality filters
    recs = recs[recs['positive_ratio'] >= 0.80]
    recs = recs[recs['actual_positive'] >= 2000]
    recs = recs.sort_values('score', ascending=False)
    
    # Output JSON
    if not recs.empty:
        results = []
        for i, (idx, row) in enumerate(recs.head(40).iterrows(), 1):
            results.append({
                'rank': i,
                'name': row['Name'],
                'positive_reviews': int(row['actual_positive']),
                'negative_reviews': int(row['actual_negative']),
                'total_reviews': int(row['total_reviews']),
                'positive_ratio': float(row['positive_ratio']),
                'match_score': float(row['score']),
                'image': row['game_image']
            })
        
        print(json.dumps({
            'status': 'success',
            'recommendations': results
        }))
    else:
        print(json.dumps({
            'status': 'error',
            'message': 'No recommendations found'
        }))
