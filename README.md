# Game Recommendation System

A hybrid machine learning recommendation engine that combines TF-IDF keyword matching with semantic embeddings to provide highly accurate game recommendations based on your library.

## 🎮 Features

- **Hybrid Similarity Engine**: Combines TF-IDF (exact tag matching) with Sentence Transformers (semantic understanding) for optimal results
- **Smart Tag Weighting**: Automatically boosts core gameplay mechanics (souls-like, metroidvania, roguelike) over artistic themes
- **Quality Filtering**: Recommends highly-rated games with substantial review counts
- **Intelligent Caching**: Automatically rebuilds model only when CSV data changes
- **Configurable Parameters**: Fine-tune diversity, quality boost, and popularity thresholds

## 📋 Requirements

### Python Version
- Python 3.11

### Required Libraries
```
pandas
numpy
scikit-learn
sentence-transformers
torch
```

## 🚀 Installation

1. **Clone or download this project**

2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn sentence-transformers torch
```

For GPU acceleration (optional but recommended)(we are using the cpu-only model of torch):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Prepare your dataset**:
   - You need to go and download the Dataset from: (https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
   - Place your `games.csv` file in the same directory as the script
   - The CSV should contain Steam game data with columns for reviews, tags, genres, descriptions, etc.

## 📊 Dataset Format

The script expects a CSV file with the following columns (note: the dataset has misaligned columns which are remapped in the code):

- **Name**: Game title
- **Supported languages** → `about_the_game`: Game description
- **Negative** → `actual_positive`: Positive review count
- **Score rank** → `actual_negative`: Negative review count
- **Screenshots** → `detailed_tags`: Detailed Steam tags
- **Tags** → `simple_genres`: Simple genre tags
- **Genres** → `steam_features`: Steam features
- **Publishers** → `developers`: Developer names
- **Categories** → `publishers`: Publisher names
- **Website** → `game_image`: Game image URL

## 💻 Usage
"The script in it's current state is made to work with a NodeJS api so feel free to adjust and use the recommend_game method and print what you want"
### Basic Usage

1. **Edit the user library** in the script (near the bottom):
```python
user_library = ["Hollow Knight", "Dark Souls", "Celeste"]
```

2. **Run the script**:
```bash
python game_recommender.py
```

3. **View results**: The script outputs top recommendations with review stats and match scores

### First Run

On the first run, the script will:
1. Load and process the CSV data (~5-10 seconds)
2. Build TF-IDF vectors (~2-3 seconds)
3. Generate semantic embeddings (~2-5 minutes for 10,000+ games)
4. Cache everything for future use

### Subsequent Runs

Once cached, the script loads in **under 1 second** and provides instant recommendations!

## ⚙️ Configuration

### Key Parameters in `recommend_games()`:

```python
recs = recommend_games(
    user_library,           # List of games you own
    top_k=80,              # Number of candidates to consider
    diversity_penalty=0.0,  # 0.0 = no diversity, 1.0 = max diversity
    quality_boost=0.5,      # How much to boost high-rated games (0.0-1.0)
    popularity_threshold_boost=True  # Extra boost for highly popular + rated games
)
```

### Quality Filters (applied after recommendations):

```python
recs = recs[recs['positive_ratio'] >= 0.80]  # Minimum 80% positive reviews
recs = recs[recs['actual_positive'] >= 2000]  # Minimum 2000 positive reviews
```

### Minimum Reviews Threshold:

```python
MIN_REVIEWS = 800  # Only consider games with 800+ total reviews
```

## 🔧 Advanced Configuration

### Adjusting Similarity Weights:

```python
TFIDF_WEIGHT = 0.5      # Weight for keyword matching
EMBEDDING_WEIGHT = 0.5  # Weight for semantic similarity
```

**Recommendations:**
- Increase `TFIDF_WEIGHT` for more tag-accurate recommendations
- Increase `EMBEDDING_WEIGHT` for more semantically similar games

### Tag Weighting:

```python
df_filtered['tags_processed'] = df_filtered['detailed_tags'].apply(
    lambda x: preprocess_field_smart(x, weight=5, boost_core=True)
)
df_filtered['features_processed'] = df_filtered['steam_features'].apply(
    lambda x: preprocess_field(x, weight=4)
)
df_filtered['about_processed'] = df_filtered['about_the_game'].apply(
    lambda x: preprocess_field(x, weight=3)
)
```

Higher weights = more importance in similarity calculations.

### KNN Neighbors:

```python
K = 45  # Number of similar games to fetch per library game
```

Increase for more diverse recommendations, decrease for more focused results.

## 🗂️ File Structure

```
project/
│
├── game_recommender.py          # Main recommendation script
├── games.csv                    # Your Steam games dataset
├── game_embeddings.pkl          # Cached embeddings (auto-generated)
├── recommendation_model_cache.pkl  # Full model cache (auto-generated)
├── cache_metadata.json          # Cache validation metadata (auto-generated)
└── README.md                    # This file
```

## 🔄 Cache Management

### Automatic Cache Invalidation

The system automatically rebuilds when:
- CSV file is modified
- CSV file hash changes
- Cache files are deleted

### Manual Cache Rebuild

To force a cache rebuild:
```python
# Delete cache files
import os
if os.path.exists('recommendation_model_cache.pkl'):
    os.remove('recommendation_model_cache.pkl')
if os.path.exists('game_embeddings.pkl'):
    os.remove('game_embeddings.pkl')
```

Or use command-line flag (if implemented):
```bash
python game_recommender.py --rebuild-cache
```

## 📈 Performance

### Timing Benchmarks (10,000 games):

**First Run (building model):**
- CSV loading: ~3 seconds
- TF-IDF building: ~2 seconds
- Embeddings generation: ~3 minutes
- Model building: ~1 second
- **Total: ~3-4 minutes**

**Subsequent Runs (cached):**
- Model loading: <1 second
- Recommendations: <0.5 seconds
- **Total: <2 second**

## 🎯 How It Works

### 1. Data Processing
- Loads Steam games CSV
- Filters games with minimum review thresholds
- Preprocesses tags, genres, and descriptions

### 2. Feature Engineering
- **TF-IDF Vectorization**: Captures exact keyword matches
- **Semantic Embeddings**: Uses Sentence Transformers to understand context
- **Smart Tag Weighting**: Boosts gameplay mechanics over themes

### 3. Hybrid Similarity
- Combines TF-IDF (50%) + Embeddings (50%)
- Normalizes both for fair comparison
- Builds KNN model on hybrid features

### 4. Recommendation Generation
- Finds similar games for each library game
- Aggregates similarity scores
- Applies quality and popularity boosts
- Filters by diversity (optional)

### 5. Quality Ranking
- Balances similarity with review quality
- Boosts highly-rated popular games
- Returns top-ranked recommendations

## 🐛 Troubleshooting

### "Game not found" error
- Check game name spelling
- Try partial names (e.g., "Dark Souls" instead of "Dark Souls: Remastered")
- Check if game exists in your CSV

### Slow embedding generation
- **Normal**: First run takes 2-5 minutes for embeddings
- **Solution**: Be patient, it only happens once
- **Optimization**: Use GPU if available (automatically detected)

### Cache not invalidating
- Check file permissions
- Manually delete `.pkl` cache files
- Verify CSV modification timestamp

## 📚 Algorithm Details

### Core Gameplay Tags
The system recognizes and boosts these gameplay mechanics:
- souls-like, metroidvania, roguelike/roguelite
- platformer, action RPG, turn-based
- hack and slash, bullet hell
- stealth, tactical, strategy, puzzle, horror

### Quality Score Formula
```
quality_score = (positive_ratio * 0.6) + (log_popularity * 0.4)
final_score = base_similarity * (1 + quality_boost * quality_score^1.5)
```

### Popularity Boost Tiers
- 100k+ reviews, 90%+ positive → 1.8x boost
- 50k+ reviews, 90%+ positive → 1.6x boost
- 20k+ reviews, 90%+ positive → 1.4x boost
- 10k+ reviews, 90%+ positive → 1.2x boost

## 🤝 Contributing

Feel free to fork and improve! Areas for enhancement:
- Add collaborative filtering
- Implement user preference learning
- Add genre-specific recommendation modes
- Create web API endpoint
- Build interactive UI

## 📝 License

This project is open source. Use freely for personal or commercial projects.

## 🙏 Acknowledgments

- **Sentence Transformers**: For semantic embeddings
- **scikit-learn**: For TF-IDF and KNN implementations
- **Steam**: For game data and tags

---

**Last Updated**: October 2025  
**Version**: 1.0

For questions or issues, please open an issue on GitHub or contact the maintainer (me).
