import numpy as np

#defining the path to our embeddings file
embeddings_file_path = r"C:\Users\Anuz\OneDrive\Desktop\excel work\Embeddings.txt"

#initializing an empty dictionary to store embeddings
embeddings = {}

#reading the embeddings from the file
with open(embeddings_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()  #splitting the line into word and vector components
        word = parts[0]  #The first part is the word
        vector = np.array(parts[1:], dtype=float)  #Converting the rest to a NumPy array of floats
        embeddings[word] = vector  #storing in the dictionary

print(f"Loaded {len(embeddings)} word embeddings.")

#creating a vocabulary mapping from words to indices
word_to_index = {word: idx for idx, word in enumerate(embeddings.keys())}
index_to_word = {idx: word for word, idx in word_to_index.items()}

print(f"Vocabulary size: {len(word_to_index)}")
vocab_dict = word_to_index 

#implementing the textrank algorithm
from sklearn.metrics.pairwise import cosine_similarity

def nepali_stemmer(word):
    """Rule-based stemmer for Nepali words"""
    suffixes = [   
     'अर्को', 'बाट', 'बाहेक', 'बाहिर', 'बाहिरपट्टी',
    'भित्र', 'का', 'करिब', 'को', 'छ', 'छिन्',
    'जोड', 'ले', 'लागि',
    'लाई', 'माथि', 'मन्तिर', 'मा', 'नजिक',
    'पछाडि', 'पहिला', 'पारि', 'प्रति', 'र',
    'संग','सहित','तल','तर','तिर',
    'तर्फ','उपर','विपरित','वरिपरि','भित्र','बिचमा']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word
def get_sentence_vector(sentence, embeddings):
    tokens = [nepali_stemmer(word) for word in sentence.split()]#tokenize the sentence into words
    vectors = [embeddings[word] for word in tokens if word in embeddings]#also appointing the vector values if the word matches 
    
    #returning a zero vector if no valid tokens are found
    if not vectors:  
        return np.zeros(100) #return a zero vector of embedding size or dimension (100)
    
    return np.mean(vectors, axis=0)  # Average out the vectors

def textrank(sentences, embeddings):
    if not sentences:  # Edge case: empty input
        return []
    
    # Calculating 30% of sentences with min 2, max 10
    num_sentences = len(sentences)
    top_n = round(num_sentences * 0.3)
    top_n = max(2, min(10, top_n))  #between 2-10

    sentence_vectors = []
    for sentence in sentences:
        vector = get_sentence_vector(sentence, embeddings)
        sentence_vectors.append(vector)

    try:
        sentence_vectors = np.array(sentence_vectors)
    except ValueError as e:
        return sentences[:top_n]  # Fallback for vectorization errors

    # Edge case: handle all-zero vectors
    if np.all(sentence_vectors == 0):
        return sentences[:top_n]

    similarity_matrix = cosine_similarity(sentence_vectors)
    scores = np.sum(similarity_matrix, axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1]]
    
    return ranked_sentences[:top_n]