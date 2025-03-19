"""
Pandas DataFrame Manipulation with Palmer Penguins Dataset
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

def load_csv(inputfile):
    """
    Load the csv as a pandas data frame
    
    Parameters
    ----------
    inputfile : string
        filename of the csv to load

    Returns
    -------
    csvdf : pandas.DataFrame
        return the pandas dataframe with the contents
        from the csv inputfile
    """
    # TODO: Implement this function
    return None


def remove_na(inputdf, colname):
    """
    Remove the rows in the dataframe with NA as values 
    in the column specified.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to check and remove rows with NA

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    # TODO: Implement this function
    return None


def onehot(inputdf, colname):
    """
    Convert the column in the dataframe into a one hot encoding.
    The newly converted columns should be at the end of the data
    frame and you should also drop the original column.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to one-hot encode

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    # TODO: Implement this function
    return None


def to_numeric(inputdf):
    """
    Extract all the 
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe

    Returns
    -------
    outputnp : numpy.ndarray
        return the numeric contents of the input dataframe as a 
        numpy array
    """
    # TODO: Implement this function
    return None


def main():
    # Load data
    df = load_csv("data/penguins.csv")

    # Remove NA
    df = remove_na(df, "species")

    # One hot encoding
    df = onehot(df, "species")

    # Convert to numeric
    df_np = to_numeric(df)

if __name__ == "__main__":
    main()
