import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    X = data.drop(["diagnosis"], axis = 1)
    y = data["diagnosis"]

    # Sclae the data
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Split the train, test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler




def get_clean_data():
    data = pd.read_csv("C:/Users/kaosh/Desktop/All_project/Side-project-ML/github/streamlit_breast_cancer/data.csv")
    
    data = data.drop(["id", "Unnamed: 32"], axis = 1)
    data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})

    
    return data

def main():
    data = get_clean_data()
    
    #train(model)
    model, scaler = create_model(data)

    # Export model

    with open('C:/Users/kaosh/Desktop/All_project/Side-project-ML/github/streamlit_breast_cancer/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('C:/Users/kaosh/Desktop/All_project/Side-project-ML/github/streamlit_breast_cancer/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    #evaluate(model)

if __name__ == '__main__':
    main()