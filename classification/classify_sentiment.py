from sklearn.dummy import DummyClassifier
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE_DIR, 'evaluation', 'evaluation_dataset.xlsx')
CORPUS_PATH = os.path.join(BASE_DIR, '..', 'preprocessing', 'Temp_preprocessed_data.csv')
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

eval_df = pd.read_excel(EVAL_PATH)

eval_df['text'] = (eval_df['summary'].fillna('') + ' ' + eval_df['content'].fillna('')).str.strip()

eval_df['ground_truth_polarity'] = (
    eval_df['ground_truth_polarity']
    .astype(str)
    .str.strip()
    .str.upper()
)

# encode labels
eval_df['polarity'] = eval_df['ground_truth_polarity'].map({
    'POSITIVE': 1,
    'NEGATIVE': 0,
    'NEUTRAL': 2
})

# filter only polarity labels (exclude neutral)
df_pol = eval_df[eval_df['polarity'] != 2]


print("Evaluation dataset size:", len(df_pol))
# Show polarity distribution with clear labels
label_map = {1: 'POSITIVE', 0: 'NEGATIVE'}
gt_counts = df_pol['polarity'].map(label_map).value_counts()
print("Ground Truth Polarity Distribution:")
print(gt_counts.to_string())

X_pol = df_pol['text']
y_pol = df_pol['polarity']

X_train_pol, X_test_pol, y_train_pol, y_test_pol = train_test_split(
    X_pol, y_pol, test_size=0.2, stratify=y_pol, random_state=42
)

# vectorize
vec_pol = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec_pol = vec_pol.fit_transform(X_train_pol)

# train model
clf_pol = LogisticRegression(max_iter=200, class_weight='balanced')
clf_pol.fit(X_train_vec_pol, y_train_pol)

# evaluate
X_test_vec_pol = vec_pol.transform(X_test_pol)

# Print ML Polarity Detection Report
ml_report = classification_report(y_test_pol, clf_pol.predict(X_test_vec_pol), digits=3, output_dict=True)
ml_report_df = pd.DataFrame(ml_report).transpose()
ml_report_df = ml_report_df.rename(index={'0': 'NEGATIVE', '1': 'POSITIVE'})
print("\nML Polarity Detection Report:")
print(ml_report_df.to_string(float_format="{:.2f}".format))

# DummyClassifier random baseline report
dummy_clf = DummyClassifier(strategy='uniform', random_state=99)
dummy_clf.fit(X_train_vec_pol, y_train_pol)
dummy_preds = dummy_clf.predict(X_test_vec_pol)
dummy_report = classification_report(y_test_pol, dummy_preds, digits=3, output_dict=True)
dummy_report_df = pd.DataFrame(dummy_report).transpose()
dummy_report_df = dummy_report_df.rename(index={'0': 'NEGATIVE', '1': 'POSITIVE'})
print("\nDummyClassifier (Random) Baseline Report:")
print(dummy_report_df.to_string(float_format="{:.2f}".format))

# Save DummyClassifier report as styled table image
fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.axis('off')
tbl2 = ax2.table(
    cellText=dummy_report_df.round(2).values,
    colLabels=dummy_report_df.columns,
    rowLabels=dummy_report_df.index,
    loc='center',
    cellLoc='center',
    colColours=['#7a3b2e']*len(dummy_report_df.columns),
    rowColours=['#7a3b2e']*len(dummy_report_df.index)
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(12)
tbl2.scale(1.3, 1.3)
for (row, col), cell in tbl2.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#7a3b2e')
    elif row % 2 == 0:
        cell.set_facecolor('#f7e6e0')
    else:
        cell.set_facecolor('#f2d6c2')
tbl2.auto_set_column_width(col=list(range(len(dummy_report_df.columns))))
plt.title('DummyClassifier (Random) Baseline Report', fontsize=16, pad=18, weight='bold', color='#7a3b2e')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'dummy_classifier_report.png'), bbox_inches='tight', dpi=200)
plt.close(fig2)

# Side-by-side comparison plot for both reports 
metrics = ['precision', 'recall', 'f1-score']
labels = ['NEGATIVE', 'POSITIVE']
ml_scores = ml_report_df.loc[labels, metrics].astype(float)
dummy_scores = dummy_report_df.loc[labels, metrics].astype(float)

fig3, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
ml_scores.plot(kind='bar', ax=axes[0], title='ML Model', color=['#2d415b', '#4f81bd', '#a6bddb'])
dummy_scores.plot(kind='bar', ax=axes[1], title='DummyClassifier', color=['#7a3b2e', '#c97b63', '#f2d6c2'])
for ax in axes:
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
fig3.suptitle('Comparison of ML Model vs. DummyClassifier', fontsize=18, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(BASE_DIR, 'results', 'ml_vs_dummy_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(fig3)

# Comparison summary
print("\nComparison of ML Model vs. DummyClassifier (Random Baseline):")
print("ML Model Accuracy: {:.2f}".format(ml_report['accuracy']))
print("DummyClassifier Accuracy: {:.2f}".format(dummy_report['accuracy']))

# Save the report as a styled table image with improved design
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')
tbl = ax.table(
    cellText=ml_report_df.round(2).values,
    colLabels=ml_report_df.columns,
    rowLabels=ml_report_df.index,
    loc='center',
    cellLoc='center',
    colColours=['#2d415b']*len(ml_report_df.columns),
    rowColours=['#2d415b']*len(ml_report_df.index)
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 1.3)
# Style header
for (row, col), cell in tbl.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2d415b')
    elif row % 2 == 0:
        cell.set_facecolor('#f2f2f2')
    else:
        cell.set_facecolor('#e6e6e6')
tbl.auto_set_column_width(col=list(range(len(ml_report_df.columns))))
plt.title('ML Polarity Detection Report', fontsize=16, pad=18, weight='bold', color='#2d415b')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'ml_polarity_detection_report.png'), bbox_inches='tight', dpi=200)
plt.close(fig)

# save model
joblib.dump(vec_pol, 'results/tfidf_polarity.joblib')
joblib.dump(clf_pol, 'results/model_polarity.joblib')

# Only ML-based sentiment prediction
corpus = pd.read_csv(CORPUS_PATH)
# Fill missing processed_text with empty string to avoid vectorizer errors
corpus['processed_text'] = corpus['processed_text'].fillna('')

# ml prediction
corpus['polarity_ml'] = clf_pol.predict(
    vec_pol.transform(corpus['processed_text'])
)

corpus['polarity_ml'] = corpus['polarity_ml'].map({
    1: 'POSITIVE',
    0: 'NEGATIVE'
})

# save output
corpus.to_csv('results/corpus_with_sentiment.csv', index=False)
print("\nClassification complete!")

