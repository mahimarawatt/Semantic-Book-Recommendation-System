# 📚 Semantic Book Recommender

A semantic book recommendation system that suggests books based on the **meaning** of your query — not just keyword matching. Built with HuggingFace sentence transformers, Chroma vector database, and a Gradio web interface.

---

## 🖥️ Demo

Enter a natural language description like *"a story about forgiveness"* or *"a book to teach children about nature"* and get back 16 relevant book recommendations with covers and descriptions.

---

## 🗂️ Project Structure

```
book_recommendation/
├── data/
│   ├── books_with_categories.csv   # Dataset with simple_categories column
│   └── tagged_description.txt      # ISBN + description per line for vector search
├── data-exploration.ipynb          # Data cleaning and preprocessing
├── text-classification.ipynb       # Zero-shot category classification
├── vector-search.ipynb             # Vector embeddings and semantic search
├── gradio-dashboard.py             # Gradio web app
└── requirements.txt
```

---

## ⚙️ How It Works

### 1. Data Exploration & Cleaning (`data-exploration.ipynb`)
- Loaded a dataset of 7000+ books with metadata (title, authors, description, categories, ratings)
- Removed books with missing descriptions, ratings, page counts, or published year
- Filtered out books with fewer than 25 words in their description
- Created a `tagged_description` field combining ISBN and description for vector indexing
- Exported cleaned data to `books_cleaned.csv` and `tagged_description.txt`

### 2. Text Classification (`text-classification.ipynb`)
- Mapped existing categories into simplified labels: `Fiction`, `Nonfiction`, `Children's Fiction`, `Children's Nonfiction`
- Used **zero-shot classification** with `cross-encoder/nli-deberta-v3-small` to predict categories for books with unknown categories
- Achieved ~70% accuracy on Fiction vs Nonfiction classification (without any training data)
- Saved the final dataset with categories to `books_with_categories.csv`

### 3. Vector Search (`vector-search.ipynb`)
- Loaded `tagged_description.txt` using LangChain's `TextLoader`
- Split into one document per book using `CharacterTextSplitter`
- Generated embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stored embeddings in a **Chroma** vector database
- Implemented `retrieve_semantic_recommendations()` to find the top-k semantically similar books for any query

### 4. Gradio Dashboard (`gradio-dashboard.py`)
- Built an interactive web UI using Gradio
- Users enter a natural language query and select a category filter
- The app retrieves and displays 16 book recommendations with thumbnails and captions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/book-recommendation.git
cd book-recommendation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the notebooks in order
```
1. data-exploration.ipynb
2. text-classification.ipynb
3. vector-search.ipynb
```

### Launch the app
```bash
python gradio-dashboard.py
```

Then open your browser at `http://localhost:7860`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data cleaning and manipulation |
| `seaborn`, `matplotlib` | Exploratory data analysis |
| `transformers` | Zero-shot text classification |
| `sentence-transformers` | Text embeddings |
| `LangChain` | Document loading and splitting |
| `Chroma` | Vector database |
| `Gradio` | Web interface |
| `scikit-learn` | Model evaluation |

---

## 📊 Model Details

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Classification model:** `cross-encoder/nli-deberta-v3-small` (zero-shot)
- **Vector store:** ChromaDB (in-memory)
- **Similarity metric:** Cosine similarity

---

## 📌 Notes
- this version uses free HuggingFace models
- Zero-shot classification achieves ~70% accuracy which is reasonable without any fine-tuning

---

## 📄 License

MIT License
