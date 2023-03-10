import preprocessing

if __name__ == __main__:
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    train_df, test_df = preprocessing_pipeline(train, test, len(train))