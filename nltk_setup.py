import nltk
nltk.data.path.append("nltk_data")

resources = ["punkt", "punkt_tab", "stopwords"]

for r in resources:
    nltk.download(r, download_dir="nltk_data")
