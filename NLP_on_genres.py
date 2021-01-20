import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def is_there_nan_values(df):
    """
    Check if there are NaN values in Dataframe
    :param df: Dataframe to analyze
    :return: True if NaN values encounted else False
    """
    return False if np.sum(df.isnull().sum()) == 0 else True


def getKeysByValue(dictOfElements, valueToFind):
    """
    Return the keys in a dictionary corresponding the values given in parameters
    :param dictOfElements: dictionary to analyze
    :param valueToFind: value to find corresponding IDs
    :return: list of corresponding IDs
    """

    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys


def transform_genres(df, remove_genres_column=False):
    """
    TODO
    :param df:
    :param remove_genres_column:
    :return:
    """
    if is_there_nan_values(df):
        raise Exception("Sorry, please remove NaN values before transforming Dataframe.")

    # We have to remove this genre string '[]' because the string encoder considers it as an empty array
    # and it raises an Exception so we drop the line to avoid issues
    df = df[df.genres != '[]']

    # columns = df.columns.values
    # features = np.delete(columns, np.argwhere(columns == target))

    genres = np.array(df.genres)

    # lexicon will contain each words in "genres" column
    lexicon = list()

    for genre in genres:
        for mot in genre.split(sep=' '):
            lexicon.append(mot)

    lexicon = np.array(lexicon)
    # we keep one time each unique word
    unique = np.unique(lexicon)

    vectorizer = CountVectorizer(strip_accents='unicode')

    # tokenize and build vocab
    vectorizer.fit(unique)

    transformation = vectorizer.transform(genres).toarray()

    col_names = list()
    for i in range(len(vectorizer.vocabulary_)):
        col_names.append(getKeysByValue(vectorizer.vocabulary_, i)[0])

    # Creation of a new dataframe with genre encoded in columns
    df_genres = pd.DataFrame(transformation, columns=col_names)

    # Adding column "genres" to merge after with the dataframe df
    df_genres.insert(0, "genres", genres)

    # Merging the original datafram and the new one created

    df_final = df.merge(df_genres, on="genres", how='inner')

    if remove_genres_column:
        df_final = df_final.drop('genres', axis=1)

    return df_final
