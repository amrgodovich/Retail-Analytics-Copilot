import nltk
# Explicitly download the 'punkt' resource
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading punkt: {e}")

# Download any other resources your code uses
try:
    nltk.download('stopwords')
except Exception as e:
    print(f"Error downloading stopwords: {e}")

print("NLTK data setup complete.")