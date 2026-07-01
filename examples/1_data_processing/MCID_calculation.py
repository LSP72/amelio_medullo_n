def calculate_MCID(df: pd.DataFrame, test: dict):
    """Function to calculate the MCID for a given test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be cleaned.
    test : str
        Name of the test for which to calculate the MCID.

    Returns
    -------
    pd.dataframe
        df with the MCID column.
    """
    MCID = Calculus.calculate_MCID(df[test + "_m_pre"], df[test + "_m_post"], threshold=30)
    df["MCID"] = MCID

    print("MCID has been calculated and added to the dataframe.")

    return df
