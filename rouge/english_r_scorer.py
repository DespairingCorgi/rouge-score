from rouge_score import rouge_scorer
import pandas as pd
# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


df = pd.read_csv('영어 - 시트2.csv')
references = df["human"].values.tolist()
candidates = df["google"].values.tolist()

scores_list = []

for ref, cand in zip(references, candidates):
    scores = scorer.score(ref, cand)
    precision = {
        "rouge1": scores['rouge1'].precision,
        "rouge2": scores['rouge2'].precision,
        "rougeL": scores['rougeL'].precision
    }
    scores_list.append(precision)

scores_df = pd.DataFrame(scores_list)
scores_df.to_csv('rouge_scores_eng_google.csv', index=False)

print(scores_df.head())