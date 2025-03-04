import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# Different values of max_depth to try
# We choose sensible values for max_depth based on the complexity of the text classification task
# and the size of our dataset (4,187 tweets in total). Shallower trees (e.g., max_depth=3) provide
# a simple baseline model, while deeper trees (e.g., max_depth=7, 10, 15) allow the model to capture 
# more complex patterns in the data. We aim to balance complexity and generalization, and avoid overfitting 
# by validating these depths using a separate validation set.
max_depth_values = [3, 5, 7, 10, 15]

def load_data():
    # Get the current working directory (repository folder)
    repo_path = os.getcwd()  # Gets the current working directory
    
    # Construct the full file paths
    exists_file = os.path.join(repo_path, 'data/h1_data/exists_climate.csv')
    dne_file = os.path.join(repo_path, 'data/h1_data/DNE_climate.csv')
    
    # Check if both files exist
    if not os.path.exists(exists_file) or not os.path.exists(dne_file):
        raise FileNotFoundError(f"One or both CSV files not found in {repo_path}. Please make sure 'exists_climate.csv' and 'DNE_climate.csv' are located in the repository folder.")
    
    # Load the CSV files
    exists_df = pd.read_csv(exists_file)
    dne_df = pd.read_csv(dne_file)
    
    # Preprocessing
    exists_df['label'] = 1  # Climate change asserting
    dne_df['label'] = 0     # Climate change denying
    
    # Combine the two datasets
    combined_df = pd.concat([exists_df, dne_df], ignore_index=True)
    
    # Vectorizing the tweet text using CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(combined_df['tweet'])
    y = combined_df['label']
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

def select_tree_model(X_train, X_val, y_train, y_val, max_depth_values):
    """
    Trains Decision Tree models for different max_depth values and criteria,
    and returns the model with the best validation accuracy.
    
    Parameters:
    X_train: Training data (features)
    y_train: Training labels
    X_val: Validation data (features)
    y_val: Validation labels
    max_depth_values: List of max_depth values to test
    
    Returns:
    best_hyperparameters: A dictionary with the best criterion and max_depth
    """
    
    # Split criteria: Gini and Entropy (Information Gain)
    criteria = ['gini', 'entropy']
    
    # Variable to store the best hyperparameters and accuracy
    best_accuracy = 0
    best_hyperparameters = {'criterion': None, 'max_depth': None}
    
    # Loop over each combination of max_depth and criterion
    for criterion in criteria:
        for max_depth in max_depth_values:
            # Train the DecisionTreeClassifier
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
            clf.fit(X_train, y_train)
            
            # Validate the model using clf.score (accuracy on validation set)
            accuracy = clf.score(X_val, y_val)
            
            # Print the result for each combination
            print(f"Criterion: {criterion}, Max Depth: {max_depth}, Validation Accuracy: {accuracy:.4f}")
            
            # Check if this is the best accuracy so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters['criterion'] = criterion
                best_hyperparameters['max_depth'] = max_depth
    
    # Print the best hyperparameters found
    print(f"Best Hyperparameters: Criterion={best_hyperparameters['criterion']}, Max Depth={best_hyperparameters['max_depth']}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    
    return best_hyperparameters

def select_best_model(X_train, X_val, X_test, y_train, y_val, y_test, best_hyperparameters):
    # Extract the best hyperparameters
    best_criterion = best_hyperparameters['criterion']
    best_max_depth = best_hyperparameters['max_depth']
    
    # Train the best model using the training data
    best_clf = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_max_depth, random_state=42)
    best_clf.fit(X_train, y_train)
    
    # Evaluate on the test set
    test_accuracy = best_clf.score(X_test, y_test)
    print(f"Test Accuracy with best hyperparameters: {test_accuracy:.4f}")
    
    return best_clf

def visualize_first_two_layers(clf, vectorizer, class_names):
    """
    Visualizes the first two layers of the decision tree and prints the decision splits with actual feature names.
    
    Parameters:
    clf : DecisionTreeClassifier
        The trained decision tree model.
    vectorizer : CountVectorizer or similar
        The vectorizer used for transforming text data into features.
    class_names: List
        List containing the class names (e.g., ['DNE', 'exists']).
    """
    # Get the feature names from the vectorizer (these are the words in your vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Option 1: Visualize as a plot with actual feature names and class names
    plt.figure(figsize=(12, 8))
    plot_tree(clf, max_depth=2, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
    plt.title("Decision Tree (First Two Layers)")
    plt.show()
    
    # Option 2: Print as text
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    print("First Two Layers of the Decision Tree:")
    
    def recurse(node, depth):
        if depth <= 1:
            if (children_left[node] != children_right[node]):  # not a leaf node
                # Get the actual feature name using the feature index
                feature_name = feature_names[feature[node]] if feature[node] != -2 else "Leaf"
                # Get the predicted class based on the majority class
                predicted_class = class_names[np.argmax(value[node])]
                print(f"Node {node}: (Feature '{feature_name}', Threshold {threshold[node]:.4f}, Class = {predicted_class})")
                print(f" --> Left child: Node {children_left[node]}")
                print(f" --> Right child: Node {children_right[node]}")
                recurse(children_left[node], depth + 1)
                recurse(children_right[node], depth + 1)

    # Start recursion from the root (node 0)
    recurse(0, 0)

def compute_entropy(y):
    # Calculate the entropy of a list of labels y
    counts = Counter(y)
    total = len(y)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * np.log2(p)
    return entropy

def compute_information_gain(X_train, y_train, keyword_index):
    """
    Computes the information gain for a split on the presence of a keyword.
    
    Parameters:
    X_train (sparse matrix): The training data features (tweets vectorized using CountVectorizer)
    y_train (array): The training data labels (asserting = 1, denying = 0)
    keyword_index (int): The index of the keyword to split on
    
    Returns:
    float: The information gain I(Y, x_i) for the keyword
    """
    # Step 1: Compute the overall entropy H(Y)
    H_Y = compute_entropy(y_train)
    
    # Step 2: Split the data based on the presence of the keyword
    keyword_present = X_train[:, keyword_index].toarray().flatten() > 0  # True if keyword appears
    keyword_absent = ~keyword_present  # True if keyword does not appear
    
    # Step 3: Compute entropy for both subsets
    y_present = y_train[keyword_present]
    y_absent = y_train[keyword_absent]
    
    # Handle case where all examples are on one side of the split
    if len(y_present) == 0 or len(y_absent) == 0:
        return 0.0
    
    # Entropy of both splits
    H_Y_given_present = compute_entropy(y_present)
    H_Y_given_absent = compute_entropy(y_absent)
    
    # Step 4: Compute the weighted conditional entropy H(Y | x_i)
    n = len(y_train)
    H_Y_given_Xi = (len(y_present) / n) * H_Y_given_present + (len(y_absent) / n) * H_Y_given_absent
    
    # Step 5: Compute the information gain I(Y, x_i)
    information_gain = H_Y - H_Y_given_Xi
    
    return information_gain

def report_information_gain(X_train, y_train, vectorizer, top_feature_index, other_keywords):
    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Compute information gain for the topmost split (top_feature_index)
    top_feature_name = feature_names[top_feature_index]
    info_gain_top = compute_information_gain(X_train, y_train, top_feature_index)
    print(f"Information Gain for topmost split '{top_feature_name}': {info_gain_top:.4f}")
    
    # Debugging: Ensure other keywords are processed
    print("Processing other keywords...")

    # Compute information gain for other selected keywords
    for keyword in other_keywords:
        print(f"Checking keyword: '{keyword}'")
        if keyword in feature_names:
            # Get the index of the keyword in the feature names
            keyword_index = np.where(feature_names == keyword)[0][0]
            info_gain_keyword = compute_information_gain(X_train, y_train, keyword_index)
            print(f"Information Gain for keyword '{keyword}': {info_gain_keyword:.4f}")
        else:
            print(f"Keyword '{keyword}' not found in the vocabulary.")

def select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # Initialize variables to store errors and accuracies
    training_errors = []
    validation_errors = []
    k_values = range(1, 21)  # k values from 1 to 20
    best_k = 1
    best_validation_accuracy = 0
    best_model = None

    # Loop through k values and train the KNN classifier for each value
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Compute training accuracy
        train_accuracy = accuracy_score(y_train, knn.predict(X_train))
        train_error = 1 - train_accuracy
        training_errors.append(train_error)

        # Compute validation accuracy
        val_accuracy = accuracy_score(y_val, knn.predict(X_val))
        val_error = 1 - val_accuracy
        validation_errors.append(val_error)

        # Update the best model based on validation accuracy
        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_k = k
            best_model = knn

        print(f"k={k}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Ensure the plot is generated and displayed correctly
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, training_errors, label="Training Error", marker='o')  # Added label
    plt.plot(k_values, validation_errors, label="Validation Error", marker='o')  # Added label
    
    # Set x-axis ticks to be integers
    plt.xticks(ticks=k_values)  # Ensure x-axis ticks show integer k values
    
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Error")
    plt.title("Training and Validation Errors for KNN")
    plt.legend()  # Now the legend will display the labeled lines
    plt.grid(True)

    # Save the plot and display it
    plt.savefig("knn_errors.png")  # Save the plot as an image
    plt.show()  # Ensure the plot is displayed

    # Continue execution without blocking
    print(f"Best k: {best_k} with Validation Accuracy: {best_validation_accuracy:.4f}")

    # Evaluate the best model on the test set
    test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Test Accuracy with best k={best_k}: {test_accuracy:.4f}")

    return best_model, best_k, best_validation_accuracy, test_accuracy

if __name__ == '__main__':
    try:
        # Q5a
        # Load the data
        X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()
        print("Data loaded successfully!")
        
        # Q5b 
        # Find the best hyperparameters
        results = select_tree_model(X_train, X_val, y_train, y_val, max_depth_values)

        # Q5c 
        # Use the best model and evaluate on the test set
        best_clf = select_best_model(X_train, X_val, X_test, y_train, y_val, y_test, results)

        # Get the topmost feature from the decision tree (root node)
        top_feature_index = best_clf.tree_.feature[0]
        top_feature_name = vectorizer.get_feature_names_out()[top_feature_index]
        print(f"Topmost feature (keyword): '{top_feature_name}' (index {top_feature_index})")

        # Visualize the first two layers of the best model
        class_names = ['DNE', 'exists']  # The class labels
        visualize_first_two_layers(best_clf,vectorizer,class_names)

        # Q5d
        # Report information gain for the topmost split and other keywords
        print("Calling report_information_gain...")
        other_keywords = ["climate", "hoax", "change", "global", "real"]
        report_information_gain(X_train, y_train, vectorizer, top_feature_index, other_keywords)
        
        #Q5e
        # Evaluate KNN classifier to classify between climate change asserting or denying tweets
        best_knn_model, best_k, best_val_acc, test_acc = select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test)

    except FileNotFoundError as e:
        print(e)