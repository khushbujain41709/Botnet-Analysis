# Bot vs Human Detection System

> This project uses **Machine Learning (XGBoost)** with **TF-IDF vectorization** to distinguish between **human** and **bot-generated text**.  
> The web interface is built using **Streamlit**.
---

## Project Description

The **Bot vs Human Detection System** is a text classification project designed to identify whether a given text was produced by a **human** or a **bot** (such as content bots, spam bots, follower bots, or customer service bots).

Using a **character-level n-gram model** and **XGBoost classifier**, the app captures linguistic and structural patterns that differentiate human writing from automated text.  
The system is integrated into a user-friendly **Streamlit dashboard** with both text input and Twitter data analysis capabilities.

---

## About the Dataset

Our training dataset consists of **250 carefully curated text samples** evenly distributed across **5 categories**.  
Each category represents distinct behavioral and linguistic patterns found in human or automated communication.

### Dataset Composition

| Category | Description |
|-----------|-------------|
| **Human Samples** | 50 human conversations and social media posts |
| **Content Bot Samples** | 50 automated content aggregation messages |
| **Follower Bot Samples** | 50 social media growth service promotions |
| **Spam Bot Samples** | 50 scam and promotional messages |
| **Customer Service Bot Samples** | 50 automated support responses |

---

## Tech Stack

| Category | Technology |
|-----------|-------------|
| Frontend  | Streamlit (Python Framework) |
| Backend   | Python 3 |
| ML Model  | XGBoost Classifier - Scikit Learn |
| Feature Extraction | TF-IDF Vectorizer (Character-level n-grams) |
| Data Processing | Pandas, NumPy |
| Storage | Joblib for model serialization |

---

## Technical Approach

This project uses **character-level n-gram analysis** instead of traditional word-based approaches because it:

- Captures bot-specific text structures such as underscores, symbols, and repetitive tokens  
- Is **language-agnostic**, working across diverse writing styles  
- Is **robust to vocabulary variations**, slang, and typos  

---

## Training Process

1. **Data Collection:** Curated dataset of 250 samples  
2. **Feature Engineering:** Character-level 3–5 gram extraction  
3. **Model Training:** XGBoost with multi-class classification  
4. **Validation:** 20% stratified split for testing  
5. **Deployment:** Streamlit-based web application for real-time detection and visualization  

---

## Application Pages

| Page | Description |
|------|--------------|
| **Home** | Overview of the system and its working process. |
| **Bot Detection** | Paste or upload text and get prediction results instantly. |
| **Twitter Analysis** | Bulk analysis of Twitter JSON exports with probability distributions |
| **About Project** | Learn about the dataset, training process, and model architecture. |
| **Test Examples** | Examples to test the bot detection system.|
---


## Installation

Follow these steps to run the project locally:

```bash
# 1️) Clone the repository
git clone https://github.com/<your-username>/bot-vs-human-detection.git
cd bot-vs-human-detection

# 2️) Create and activate a virtual environment (optional)
python -m venv env
source env/bin/activate      # For Linux/Mac
env\Scripts\activate         # For Windows

# 3️) Install dependencies
pip install -r requirements.txt

# 4️) Run the app
streamlit run main.py
 ```
---
### Developers

| Name | Roll Number |
|----------|----------------|
| [Khushbu Jain](https://github.com/khushbujain41709) | 23115047 |
| [Kruti Dadriwal](https://github.com/krutidadriwal) | 23115050 |
| [Mukka Manideep](https://github.com/Manideep-Mukka) | 23115058 |
| [P.Sai Madhuri Bai](https://github.com/Madhuri2829) | 23115067 |


 
---

## Acknowledgments

Special thanks to:

- [**Dr. Jitendra Kumar Rout**](https://www.nitrr.ac.in/viewdetails.php?q=cse.jkrout) for fostering innovation and creativity.  
- **Streamlit** for providing a seamless and intuitive UI framework.  
- **XGBoost** and **Scikit-learn** for enabling robust ML capabilities.  

---
