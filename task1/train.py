import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_and_save_model():
  # 1. Load dataset
  df = pd.read_csv('Titanic-Dataset.csv')

  features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
  target = 'Survived'

  X = df[features]
  y = df[target]

  # 2. Stratified split preserves class balance
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # 3. Define production preprocessing pipelines
  numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
  categorical_features = ['Pclass', 'Sex', 'Embarked']

  numeric_transformer = Pipeline(
      steps=[('imputer', SimpleImputer(strategy='median'))]
  )

  categorical_transformer = Pipeline(
      steps=[
          ('imputer', SimpleImputer(strategy='most_frequent')),
          ('onehot', OneHotEncoder(handle_unknown='ignore')),
      ]
  )

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', numeric_transformer, numeric_features),
          ('cat', categorical_transformer, categorical_features),
      ]
  )

  # 4. Combine preprocessor & model into one deployable pipeline
  model_pipeline = Pipeline(
      steps=[
          ('preprocessor', preprocessor),
          (
              'classifier',
              RandomForestClassifier(
                  n_estimators=200, max_depth=8, random_state=42
              ),
          ),
      ]
  )

  # 5. Train & evaluate
  model_pipeline.fit(X_train, y_train)
  y_pred = model_pipeline.predict(X_test)

  print(f'Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
  print('\nClassification Report:\n', classification_report(y_test, y_pred))

  # 6. Serialize the entire pipeline to disk
  joblib.dump(model_pipeline, 'titanic_model.joblib')
  print("Artifact saved successfully as 'titanic_model.joblib'")


if __name__ == '__main__':
  train_and_save_model()
