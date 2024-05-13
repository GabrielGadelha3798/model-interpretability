import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_features_custom(df, ncols, plot_types, feature_by_outcome=True, outcome_name='Outcome'):
    # List of colors
    colors = sns.color_palette('husl', n_colors=len(df.drop(outcome_name, axis=1).columns))
    features = df.drop(outcome_name, axis=1).columns  # Drop outcome column from the features list
    nrows = int(np.ceil(len(features) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 10))
    ax = ax.flatten()  # Flatten the axis array for easier indexing

    # Ensure there are enough axes
    if len(ax) > len(features):
        for extra_ax in ax[len(features):]:
            extra_ax.remove()

    for i, feature in enumerate(features):
        color = colors[i % len(colors)]
        plot_type = plot_types[i] if i < len(plot_types) else 'b'  # Default to boxplot if not specified
        
        if plot_type == 'b':  # Boxplot
            if feature_by_outcome:
                sns.boxplot(x=outcome_name, y=feature, data=df, ax=ax[i], color=color)
            else:
                sns.boxplot(y=feature, data=df, ax=ax[i], color=color)
        
        elif plot_type == 'c':  # Countplot
            if feature_by_outcome:
                sns.countplot(x=feature, hue=outcome_name, data=df, ax=ax[i], palette='husl')
            else:
                sns.countplot(x=feature, data=df, ax=ax[i], palette='husl')
        
        elif plot_type == 'h':  # Heatmap
            if feature_by_outcome:
                # Create a crosstab for the heatmap
                crosstab = pd.crosstab(df[outcome_name], df[feature])
                sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax[i])
            else:
                raise ValueError("Heatmap does not support non-outcome feature plotting directly.")

        elif plot_type == 'r':  # Barplot
            if feature_by_outcome:
                group_data = df.groupby(outcome_name)[feature].mean()
                group_data.plot(kind='bar', ax=ax[i], color=color)
                ax[i].set_ylabel('Mean ' + feature)
            else:
                df[feature].value_counts().plot(kind='bar', ax=ax[i], color=color)
        
        ax[i].set_title(feature)

    plt.tight_layout()
    plt.show()



def plot_features_boxplot(df, ncols,feature_by_outcome=True,violinplot=False,outcome_name = 'Outcome'):
    # List of colors
    colors = sns.color_palette('husl', n_colors=len(df.drop(outcome_name,axis=1).columns))
    nrows = int(np.ceil(len(df.drop(outcome_name,axis=1).columns) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 10))
    for i, feature in enumerate(df.drop(outcome_name,axis=1).columns):
        # one color for each plot
        color = colors[i % len(colors)]
        if feature_by_outcome:
            if violinplot:
                sns.violinplot(x=outcome_name, y=feature, data=df, ax=ax[i // ncols, i % ncols], color=color)
            else:
                sns.boxplot(x=outcome_name, y=feature, data=df, ax=ax[i // ncols, i % ncols], color=color)
        else:
            if violinplot:
                sns.violinplot(y=feature, data=df, ax=ax[i // ncols, i % ncols], color=color)
            else:
                sns.boxplot(y=feature, data=df, ax=ax[i // ncols, i % ncols], color=color)       
        ax[i // ncols, i % ncols].set_title(feature)
    plt.tight_layout()
    plt.show()

def calculate_metrics(y_test, y_pred,round_digits = 2):
    """
    calculate the accuracy, precision, recall and f1 score for a given set of predictions and true labels
    Rounds values to the N decimal place based on the value of the 'round_digits' variable
    """
    return {
        'accuracy': round(accuracy_score(y_test, y_pred),round_digits),
        'precision': round(precision_score(y_test, y_pred),round_digits),
        'recall': round(recall_score(y_test, y_pred),round_digits),
        'f1': round(f1_score(y_test, y_pred),round_digits),
        'auc': round(roc_auc_score(y_test, y_pred),round_digits)
    }

def create_metrics_df(classifiers, X_test, Y_test):
    """
    Creates a DataFrame from a list of classifiers and the test data.
    Uses calculate_metrics to calculate the metrics for each classifier.
    The classifiers list contains tuples with classifier name (0) and the classifier itself (1) the name is used for the column name in the resulting DataFrame.
    """
    metrics = {}
    for classifier in classifiers:
        metrics[classifier[0]] = calculate_metrics(Y_test, classifier[1].predict(X_test))
    return pd.DataFrame(metrics)


def plot_partial_dependence(cls, X_test):
    # Número de features
    n_features = X_test.shape[1]
    
    # Determinar o número de colunas e linhas para os subplots
    n_cols = 2  # Você pode ajustar isso dependendo de quanto espaço horizontal está disponível
    n_rows = (n_features + n_cols - 1) // n_cols  # Arredonda para cima para ter certeza que todas as features têm espaço

    # Criar uma figura com subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))  # Ajuste o tamanho conforme necessário
    axs = axs.flatten()  # Transformar a matriz de axs em um array 1D para facilitar a indexação

    # Plotar a dependência parcial para cada feature
    for i in range(n_features):
        if i < len(axs):  # Verifica se ainda há axs disponíveis
            PartialDependenceDisplay.from_estimator(
                cls, X_test, [i], ax=axs[i], kind='both')
        else:
            break  # Evitar erro se houver mais features do que axs

    # Ocultar axs extras se o número de features for menor que o número de axs
    for ax in axs[n_features:]:
        ax.set_visible(False)

    # Ajustar layout e mostrar plot
    plt.tight_layout()
    plt.show()

def plot_pdp_interactions(model, X_data, column_index, feature_names=None):
    """
    Plots two-way partial dependence plots for a specified feature in relation to all other features.

    Parameters:
    - model: The trained machine learning model.
    - X_data: The dataset (typically the test set) used for the PDP.
    - column_index: The index of the feature to analyze for interactions.
    - feature_names: List of feature names for labeling purposes. If None, indices will be used.
    """
    n_features = X_data.shape[1]
    n_rows = (n_features - 1) // 2 + (n_features - 1) % 2  # Calculate the number of rows needed
    
    # Creating a figure with appropriate size
    fig, axs = plt.subplots(n_rows, 2, figsize=(20, 5 * n_rows), squeeze=False)  # Ensure axs is always 2D

    # Exclude the main feature itself and prepare feature pairs
    feature_indices = [i for i in range(n_features) if i != column_index]
    pairings = [(column_index, i) for i in feature_indices]  # Create pairs (main_feature, other_feature)

    for idx, ax in enumerate(axs.ravel()):
        if idx < len(pairings):
            feature_pair = pairings[idx]
            if feature_names is not None:  # Changed condition to check explicitly for None
                title = f'PDP for {feature_names[feature_pair[0]]} and {feature_names[feature_pair[1]]}'
            else:
                title = f'PDP for Feature Index {feature_pair[0]} and Feature Index {feature_pair[1]}'
            disp = PartialDependenceDisplay.from_estimator(
                model, X_data, features=[feature_pair], ax=ax)
            ax.set_title(title)
        else:
            ax.set_visible(False)  # Hide unused axes

    plt.tight_layout()
    plt.show()

def transform_heart_data(data):
    

    encoding_maps = {
        'Sex': {'M': 1, 'F': 2},
        'ChestPainType': {'TA': 3, 'ATA': 2, 'NAP': 1, 'ASY': 0},
        'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
        'ExerciseAngina': {'Y': 1, 'N': 0},
        'ST_Slope': {'Up': 1, 'Flat': 0, 'Down': -1}
    }

    data = data.copy()
    for column, mapping in encoding_maps.items():
        data[column] = data[column].map(mapping)
        print(f"Encoded {column} with mapping {mapping}.")

    # Handling zeros in RestingBP and Cholesterol by replacing with median values of each HeartDisease class
    for column in ['RestingBP', 'Cholesterol']:
        median_values = data[data[column] != 0].groupby('HeartDisease')[column].median()
        print(f"Substituting {column} zeros with the medians {median_values[0]} and {median_values[1]} for group 0 and 1.")
        # Map the median values and explicitly cast them to int to avoid dtype incompatibility
        data.loc[data[column] == 0, column] = data.loc[data[column] == 0, 'HeartDisease'].map(median_values).astype(int)

    # Return the transformed DataFrame
    return data