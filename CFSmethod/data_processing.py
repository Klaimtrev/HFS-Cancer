import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import arff, loadmat

def load_and_preprocess_data(filepath):
    correct_gene_ids = None  # This will store feature indices instead of names.

    if filepath.endswith(".arff"):
        # Load ARFF dataset
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)

        # Convert target column from byte to string and encode it
        df['Tissue'] = df['Tissue'].apply(lambda x: x.decode('utf-8'))
        label_encoder = LabelEncoder()
        df['Tissue'] = label_encoder.fit_transform(df['Tissue'])

        # Split the dataset into features and target
        X = df.drop(columns=['Tissue', 'ID_REF'])  
        y = df['Tissue']

    elif filepath.endswith(".mat"):
        # Load MAT dataset
        mat_data = loadmat(filepath)
        keys = [key for key in mat_data.keys() if not key.startswith('__')]

        # Assume the first key contains features, and the second key contains labels
        X = pd.DataFrame(mat_data[keys[0]])
        y = pd.Series(mat_data[keys[1]].ravel())

        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y))

    elif filepath.endswith(".csv"):
        # Load CSV leukemia dataset
        leukemia_df = pd.read_csv(filepath)

        all_columns = [col for col in leukemia_df.columns if 'ALL' in col]
        aml_columns = [col for col in leukemia_df.columns if 'AML' in col]

        X = pd.concat([leukemia_df[all_columns], leukemia_df[aml_columns]], axis=1).values.T
        y = [0] * len(all_columns) + [1] * len(aml_columns)

        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y))  # Ensure y is a Pandas Series


    else:
        raise ValueError("Unsupported file format. Only .mat, .csv, and .arff are supported.")

    # Convert X to a DataFrame to get numerical indices
    X = pd.DataFrame(X)

    # Store feature indices instead of names
    correct_gene_ids = list(range(X.shape[1]))  # [0, 1, 2, ..., n_features-1]

    # Split dataset into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, X, label_encoder, correct_gene_ids
