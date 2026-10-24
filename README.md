# Customer Review Sentiment Analysis

A course project that collects Trustpilot reviews with Requests and Beautiful Soup, prepares a labeled dataset, compares sentiment classifiers, and serves predictions through a Streamlit app.

The app classifies review text as **negative**, **neutral**, or **positive** and displays the predicted class, its model probability, and a probability chart for all three classes.

## Project workflow

1. **Collect reviews:** `scrape.ipynb` extracts review text and star ratings from Trustpilot pages for Amazon, eBay, and other businesses.
2. **Prepare the dataset:** the notebook combines the source CSV files, records their origins, and removes duplicate review text.
3. **Process text:** `Sentiment_Prediction.ipynb` lowercases text, tokenizes it, removes English stopwords, and applies WordNet lemmatization.
4. **Train and compare models:** TF-IDF features with unigrams, bigrams, and trigrams feed Multinomial Naive Bayes, a PyTorch neural network, cross-validated Logistic Regression, and Linear SVC.
5. **Run predictions:** `app.py` loads the saved Logistic Regression model and TF-IDF vectorizer for interactive review analysis.

## Dataset

The included `trustpilot_all_sources_combined.csv` contains **8,014 reviews** with these columns:

| Column | Description |
| --- | --- |
| `text` | Review text |
| `stars` | Original rating from 1 to 5 |
| `sentiment` | Sentiment label derived from the rating |
| `source_file` | Source CSV filename |

| Star rating | Sentiment | Reviews in the combined CSV |
| --- | --- | ---: |
| 1-2 | Negative | 2,682 |
| 3 | Neutral | 2,650 |
| 4-5 | Positive | 2,682 |

These labels are derived from star ratings rather than manual annotation of the review text.

## Repository files

| File | Purpose |
| --- | --- |
| [`scrape.ipynb`](scrape.ipynb) | Scraping, labeling, and combining review CSV files |
| [`Sentiment_Prediction.ipynb`](Sentiment_Prediction.ipynb) | Text preprocessing, model training, evaluation, and export |
| [`app.py`](app.py) | Streamlit sentiment prediction interface |
| `logistic_model.joblib` | Saved Logistic Regression classifier |
| `tfidf_vectorizer.joblib` | Saved TF-IDF vectorizer |
| `trustpilot_all_sources_combined.csv` | Combined review dataset |
| `trustpilot_*_star_reviews.csv` | Individual review datasets by business and rating |

## Run the app

Install Python and Git, then clone the repository:

```sh
git clone https://github.com/menghong-heng/beautiful_soup_scrape.git
cd beautiful_soup_scrape
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the app dependencies and start Streamlit from the repository root so it can find both saved model files:

```sh
python -m pip install streamlit joblib numpy pandas scikit-learn
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, enter a review, and select **Predict Sentiment**. For example: `The delivery arrived on time and the product works well.`

## Explore the notebooks

Install the additional scraping, training, and notebook dependencies:

```sh
python -m pip install jupyter requests beautifulsoup4 nltk matplotlib torch
python -m jupyter notebook
```

Open `Sentiment_Prediction.ipynb` to explore the included dataset and model comparisons. The notebook downloads NLTK resources and uses CUDA when available. Its NLTK download path is `/tmp/nltk_data`; on Windows, change both the path entry and download directories to a writable local folder if needed.

The training notebook converts TF-IDF features into dense arrays, so training can require substantial memory. Training again also overwrites the saved model and vectorizer when the export cell runs.

Open `scrape.ipynb` only if you want to collect fresh reviews. Its scraping cells make live web requests and overwrite the corresponding CSV files; the existing dataset is enough to explore training without scraping again.

## Current limitations

- Dependency versions from the original training environment are not recorded. If the saved model cannot load with your scikit-learn version, retrain and export both artifacts together in your environment.
- The app passes review text directly to the saved vectorizer, while the training notebook also applies NLTK preprocessing. This difference can affect predictions.
- For inputs shorter than five words, the app adds two randomly selected context phrases. Repeated predictions for short inputs can therefore differ.
- Displayed confidence is the model's predicted probability, not a guarantee of correctness. Sarcasm, mixed opinions, and text unlike the training reviews can be difficult to classify.
