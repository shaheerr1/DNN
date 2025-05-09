# DNN

# Recommender System with Collaborative Filtering

This repository implements and evaluates two collaborative filtering models on the MovieLens 1M dataset:

- **MLP Recommender**: A neural collaborative filtering model using user/item embeddings and a multi-layer perceptron.
- **AutoRec**: An autoencoder-based model that reconstructs user–item interaction vectors.

---

## Project Structure

```
recommender-mlp/
├── data/                      # MovieLens dataset files
│   ├── ratings.dat            # User–item rating interactions
│   └── movies.dat             # Movie metadata (titles, genres)
├── mlp_recommender.ipynb      # Jupyter Notebook for the MLP model
├── autoencoder.ipynb          # Jupyter Notebook for the AutoRec model
├── requirements.txt           # Python package dependencies
└── README.md                  # This documentation
```

---

## Setup

1. **Clone the repository**

```bash
git clone <repo-url>
cd recommender-mlp
```

2. **Create & activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download & place MovieLens data**

- Download MovieLens‑1M from [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)
- Unzip and copy `ratings.dat` and `movies.dat` into the `data/` folder.

---

## Usage

All experiments are provided as Jupyter Notebooks. Launch Jupyter and open the desired notebook:

```bash
jupyter notebook
```

1. **MLP Recommender** (`mlp_recommender.ipynb`)

   - Imports and cleans data
   - Defines a PyTorch MLP model with user/item embeddings
   - Trains and evaluates using RMSE, MAE, HR\@k
   - Generates Top‑N movie recommendations by title

2. **AutoRec (Autoencoder)** (`autoencoder.ipynb`)

   - Builds a user–item rating matrix
   - Defines an autoencoder (AutoRec) in PyTorch
   - Trains with masked reconstruction loss on known entries
   - Evaluates with reconstruction loss, RMSE, MAE, HR\@k
   - Recommends Top‑N movies per user based on reconstructed scores

---

## Evaluation Metrics

- **Reconstruction Loss (AutoRec)** — MSE over observed ratings only
- **RMSE** — Root Mean Squared Error on held‑out test ratings
- **MAE** — Mean Absolute Error on held‑out test ratings
- **HR\@k (Hit Ratio @ k)** — Fraction of times the true test item appears in the top‑k predictions for each user

---

## Generating Recommendations

Both notebooks include helper functions:

- `recommend_movies(model, user_id, df, n)` for the MLP model
- `recommend_autoRec(user_id, R_pred, train_df, idx2title, n)` for AutoRec

They return a list of movie titles for the specified user.

---
