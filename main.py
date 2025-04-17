import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein  
import difflib
import re
from transformers import pipeline
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt

def compute_cosine_similarity(text1, text2):
    """
    Compute the cosine similarity between two texts using TF-IDF vectorization.
    """
    if not text1 or not text2:
        return 0.0
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf = vectorizer.transform([text1, text2])
    sim = cosine_similarity(tfidf[0], tfidf[1])
    return sim[0][0]

def compute_jaccard_similarity(text1, text2):
    """
    Compute the Jaccard index between two texts.
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 and not tokens2:
        return 1.0  # Both empty; treat as identical
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def compute_levenshtein_distance(text1, text2):
    """
    Compute the Levenshtein distance (edit distance) between two texts.
    """
    return Levenshtein.distance(text1, text2)

def highlight_changes(text1, text2):
    diff = list(difflib.ndiff(text1.split(), text2.split()))
    return '\n'.join(diff)


def generate_feedback(question, orig_answer, thought, rubric, num_return_sequences=3, num_beams=10):
    """
    Generate feedback and score for a given question, thought, and rubric using Flan-T5
    """
    prompt = (
        "You are a helpful teaching assistant. Create a clear, point-by-point feedback and score "
        " that evaluates the following answer using thought,question and rubric.\n\n"
        f"question:\n{question}\n\n"
        f"Original Answer:\n{orig_answer}\n\n"
        f"Thought:\n{thought}\n\n"
        f"Rubric:\n{rubric}\n\n"
        "Feedback:"
        "Score:"
    )
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=num_beams,
        early_stopping=True,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        temperature=0.9,
    )
    
    result = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return result


def generate_evaluation(prompt, generator):
    """
    Step 2: Evaluate whether feedback was addressed using BART-based model
    """
    result = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.7)
    output_text = result[0]['generated_text'].strip()
    
    try:
        evaluation = pd.read_csv("Data/input_dataset.csv")
    except Exception as e:
        new_score_match = re.search(r'(\d+)', output_text)
        new_score = int(new_score_match.group(1)) if new_score_match else None
        analysis = output_text
        evaluation = {"new_score": new_score, "analysis": analysis}
    return evaluation



# Load input training dataset
df = pd.read_csv("Data/input_dataset.csv")

# Extract features and process each sample using the Flan-T5 model
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Feedback and score"):
    question = row["question"]
    orig_answer = row["orig_answer"]
    thought = row["thought"]
    rubric = row["rubric"]
    
    feedback, score = generate_feedback(question, orig_answer, thought, rubric, num_return_sequences=3)
    
df["feedback"] = feedback
df["score"] = score

df.to_csv("Data/input_dataset.csv", index=False)

print("Feedback and scores generated and saved to input_dataset.csv.")
# Load BART model for evaluation phase
generator = pipeline("text2text-generation", model="facebook/bart-base")

new_scores = []
evaluations = []

# Iterate through submissions and compare resubmissions to feedback for scoring
for idx, row in df.iterrows():
    original_text = str(row["orig_answer"])
    feedback = str(row["feedback"])
    score = str(row["score"])
    resubmitted_text = str(row["resub_answer"])
    
    prompt = (
        f"Original Text: {original_text}\n"
        f"Feedback (rubric): {feedback}\n"
        f"Original Score: {score}\n"  
        f"Resubmitted Text: {resubmitted_text}\n\n"
        "Task:\n"
        "1. Evaluate the resubmitted text against the feedback:\n"
        "   - Provide a bullet-point list of feedback items that have been implemented.\n"
        "   - Provide a bullet-point list of feedback items that are missing or not implemented.\n"
        "2. Based on your analysis, assign a new score out of 10. "
        "This new score must not be lower than the original score. If improvements are significant, increase the score; otherwise, keep it the same.\n\n"
        "Output your response in csv format as new_score:\n"
    )
    
    evaluation_output = generate_evaluation(prompt, generator)
    
    try:
        original_score_int = int(score)
    except:
        original_score_int = 0
    
    gen_score = evaluation_output.get("new_score", None)
    if gen_score is not None:
        new_score = max(original_score_int, gen_score)
    else:
        new_score = original_score_int
    
    new_scores.append(new_score)
    evaluations.append(evaluation_output.get("analysis", ""))

df['new_score'] = new_scores
df.to_csv('Data/output.csv', index=False)
print("Updated dataset and saved with new_score column.")

# Visualize the results
# Calculating cosine similarity for the original answer and feedback
tqdm.pandas(desc="Calculating Cosine Similarities")
df['cosine_similarity'] = df.progress_apply(
    lambda row: compute_cosine_similarity(row['orig_answer'], row['feedback']),
    axis=1
)
df.to_csv("data/output.csv", index=False)

# Calculate the cosine similarity for resubmitted answers and feedback
tqdm.pandas(desc="Calculating Cosine Similarities")
df['cosine_similarity_new'] = df.progress_apply(
    lambda row: compute_cosine_similarity(row['resub_answer'], row['feedback']),
    axis=1
)

# Save the updated DataFrame with all columns to a new CSV file
df.to_csv("data/output.csv", index=False)

# Scatter plot to visualize how the resubmission scores compare to the original scores
file_path = 'Data/output.csv'  
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
plt.scatter(df['score'], df['new_score'], color='blue', alpha=0.6)
plt.title('Scatter Plot of Old Score vs New Score')
plt.xlabel('Old Score')
plt.ylabel('New Score')
plt.grid(True)
plt.show()


x = list(range(len(df)))

# Plot similarity scores
plt.figure(figsize=(14, 6))
plt.plot(x, df["cosine_similarity"], 'o-', label="Similarity score between Original answer and feedback")
plt.plot(x, df["cosine_similarity_new"], 'x-', label="Similarity score between Resubmitted answer and feedback")

# Add labels and title
plt.title("Comparison of Similarity scores")
plt.xlabel("Question")
plt.ylabel("Similarity scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# The above code is a complete script that processes a dataset of student answers, generates feedback and scores using a Flan-T5 model, and evaluates resubmissions using a BART-based model. 
# It also visualizes the relationship between old and new scores.
# The script includes functions for computing various similarity metrics, generating feedback, and evaluating the resubmissions.
# The final output is saved to a CSV file, and a scatter plot is displayed to visualize the relationship between old and new scores.


                      