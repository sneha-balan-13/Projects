# %%
import pandas as pd

# Load merged csv
df = pd.read_csv('data/merged_tickets.csv')
df.head()

# %%
# Combine Subject + Description
df['text'] = df['subject'] + ' ' + df['description']

# %%
# pip install scikit-learn

# %%
# Encode priority label
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['priority_encoded'] = le.fit_transform(df['priority_label'])
print(le.classes_)      # To see label order

# %%
# Train Test Split
from sklearn.model_selection import train_test_split

X = df['text']
y = df['priority_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# %%
# Train a classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

df['priority_label'].value_counts()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification_Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# %%
import joblib

# Save the model and vectorizer
joblib.dump(model, 'priority_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

# %%
# Alternate Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Common pipeline setup for text classification
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(class_weight='balanced',random_state=42))
])

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(class_weight='balanced',random_state=42))
])

# Train basic models
rf_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Evaluate validation set
print("Random Forest Accusracy: ", rf_pipeline.score(X_test, y_test))
print("SVM Accuracy: ", svm_pipeline.score(X_test, y_test))

# %%
from sklearn.model_selection import GridSearchCV

# Tune only Random Forest
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# %%
# Select the best model
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation F1 score:", grid_search.best_score_)

# Best trained pipeline
final_model = grid_search.best_estimator_

# Validate on test set
val_accuracy = final_model.score(X_test, y_test)
print("Validation Accuracy (Best Model): ", val_accuracy)

# %%
import joblib

# Save the best model
joblib.dump(final_model, '/workspaces/ticket-priority-app/models/final_model.pkl')

joblib.dump(X_test, '/workspaces/ticket-priority-app/models/X_test.pkl')
joblib.dump(y_test, '/workspaces/ticket-priority-app/models/y_test.pkl')

# %%
import matplotlib.pyplot as plt

# Check distribution
label_counts = y_train.value_counts()
print(label_counts)

# Optional: plot it
label_counts.plot(kind='bar', title='Class Distribution in y_train')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.show()


# %%
df['priority_label'].value_counts()


