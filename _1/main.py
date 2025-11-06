import pandas as pd
from sklearn.linear_model import LinearRegression

# Data
d = {
    'study_hrs'   : [7, 4, 5, 6],
    'sleep_hours' : [4, 9, 7, 8],
    'score_prev1' : [50, 69, 72, 70],
    'score_prev2' : [30, 70, 75, 74],
    'subject_type': ['Phy', 'Chem', 'IP', 'Maths']
}
df = pd.DataFrame(d)

# Subject numeric id
df['subject_id'] = df['subject_type'].astype('category').cat.codes

# Model: predict *score_prev2* 
X = df[['study_hrs', 'sleep_hours', 'score_prev1', 'score_prev2', 'subject_id']]
y = df['score_prev2']            # target = previous score - 2 
model = LinearRegression()
model.fit(X, y)

# Subject lookup dictionary
subject_to_code = pd.Series(df['subject_id'].values,
                            index=df['subject_type']).to_dict()

def predict(study_hours, sleep_hours, score_prev1, score_prev2, subject_type):
    subj_id = subject_to_code[subject_type]
    X_new = pd.DataFrame([[study_hours, sleep_hours, score_prev1, score_prev2, subj_id]],
                         columns=X.columns)
    return float(model.predict(X_new)[0])

# prediction
print(predict(5.5, 7.0, 85, 78, 'Chem'))

