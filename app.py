from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)


# Load the iris dataset once
iris = load_iris()
dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
dataframe['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

def get_iris_data():
    print("Loading iris data...")
    print(dataframe.head())

def visualize_iris_data():
    print("Visualizing iris data...")
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(dataframe, hue="species", palette="husl")
    plt.show()

def find_null_values():
    print("Finding missing values...")
    print(dataframe.isnull().sum())

def split_dataset():
    print("Splitting dataset...")
    X = dataframe.iloc[:, :-1]  # All rows, all columns except last
    y = dataframe['species']    # All rows, only last column which is 'species'.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    print("Training model...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Model accuracy:", knn.score(X_test, y_test))


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Generate and print confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Generate and print classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)

    # Visualize confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    get_iris_data()
    # visualize_iris_data()
    find_null_values()
    X_train, X_test, y_train, y_test = split_dataset()
    train_model(X_train, X_test, y_train, y_test)
    knn.fit(X_train, y_train)
    evaluate_model(knn, X_test, y_test)


