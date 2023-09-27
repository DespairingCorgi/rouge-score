from collections import Counter
from sklearn.metrics import f1_score, recall_score, precision_score
from konlpy.tag import Okt

tokenizer = Okt()

def get_ngrams(text, n):
    """Get n-grams from the text."""
    tokens = text.split()
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def lcs(X , Y): 
    """Get the length of the longest common subsequence."""
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]

def rouge_score(reference, candidate):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    # Tokenize the texts
    reference_tokens = tokenizer.morphs(reference)#reference.split()
    candidate_tokens = tokenizer.morphs(candidate)#candidate.split()

    # Get n-grams
    reference_1grams = get_ngrams(reference, 1)
    candidate_1grams = get_ngrams(candidate, 1)
    reference_2grams = get_ngrams(reference, 2)
    candidate_2grams = get_ngrams(candidate, 2)

    # Get LCS length
    lcs_length = lcs(reference_tokens, candidate_tokens)

    # Calculate scores
    scores = {
        'rouge1': len(set(reference_1grams) & set(candidate_1grams)) / len(set(candidate_1grams)) if candidate_1grams else 0,
        'rouge2': len(set(reference_2grams) & set(candidate_2grams)) / len(set(candidate_2grams)) if candidate_2grams else 0,
        'rougeL': lcs_length / len(candidate_tokens)
    }
    
    return scores

def get_score(ref, cand):
    scores = rouge_score(ref, cand)
    return scores

def get_scores(refs, cands):
    scores = [get_score(r, c) for r,c in zip(refs, cands)]
    return scores


import pandas as pd

df = pd.read_csv('한국어 - 시트2.csv')

references = df["human"].values.tolist()
candidates = df["matis"].values.tolist()

outputdf = pd.DataFrame(get_scores(references, candidates))

outputdf.to_csv('rouge_scores_kor_matis.csv', index=False)