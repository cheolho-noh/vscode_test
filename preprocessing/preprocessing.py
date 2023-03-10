# 데이터 분석 라이브러리 불러오기
import pandas as pd
import numpy as np
import random as rnd

def integrate_dataframe(train:pd.DataFrame, test:pd.DataFrame) -> pd.DataFrame:
    """전처리를 위해 Train, Test 데이터 통합"""
    df = pd.concat([train, test], axis=0)
    return df

def replace_null_feature(df:pd.DataFrame) -> pd.DataFrame:
    """Embarked(승선지), Fare(요금) Null 값 대체"""
    # Embarked 변수 null을 최빈값으로 대체
    df['Embarked'] = df['Embarked'].fillna(df.Embarked.dropna().mode()[0])
    # Fare 변수 null을 중앙값으로 대체
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    return df

def transform_feature_type(df:pd.DataFrame) -> pd.DataFrame:
    """Embarked, Sex 변수 Type 변환"""
    # Embarked 변수 type int로 변환
    EMBARKED_MAPPING = {"S": 0, "C": 1, "Q": 2}
    df['Embarked'] = df['Embarked'].map(EMBARKED_MAPPING).astype(int)

    # Sex 변수 type int로 변환
    SEX_MAPPING = {"female": 1, "male": 0}
    df['Sex'] = df['Sex'].map(SEX_MAPPING).astype(int)
    return df

def replace_null_age_median(df:pd.DataFrame) -> pd.DataFrame:
    """PClass, Gender 변수 조합별 Age의 Median 값으로 Null 값 대체"""
    # Sex, Pclass의 조합으로 Age를 추측하기 위한 empty array 생성하기
    guess_ages = np.zeros((2, 3))
    # Sex, Pclass의 6가지 조합에 대해 반복하여 Age의 추측값 계산하기 -> 조합별 중앙값 이용
    for i in range(2):
        for j in range(3):
            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna()
            # Sex가 i, Pclass가 j+1인 Age를 추출하고 결측값은 제외
            age_med = guess_df.median()
            guess_ages[i, j] = int(age_med / 0.5 + 0.5) * 0.5
    # Age가 Null인 데이터 중앙값으로 대체
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), 'Age'] = guess_ages[i, j]
    # int로 변환
    df['Age'] = df['Age'].astype(int)
    return df

def make_derived_feature(df:pd.DataFrame) -> pd.DataFrame:
    """파생변수 Title, AgeBand, FamilySize, IsAlone, Age*Class, FareBand 생성"""
    # 호칭변수 Title 생성
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)
    # Name Column에서 Master, Miss, Mr, Mrs 4개의 호칭 추출 후 나머지는 Rare로 분류
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer',
                                       'Dona'], 'Rare')
    # Mlle을 Miss로 변경
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    # Ms를 Miss로 변경
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    # Mme를 Mrs로 변경
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # Title 변수 ordinal 변수로 변환
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    # Title이 없으면 0번으로 분류
    df['Title'] = df['Title'].fillna(0)

    # 연령대 변수 AgeBand를 ordinal(0,1,2,3,4)하게 생성
    df['AgeBand'] = df['Age'].apply(lambda x: 0 if x <= 16 else (1 if x > 16 and x <= 32
                                                                 else (2 if x > 32 and x <= 48
                                                                       else (3 if x > 48 and x <= 64 else 4))))

    # 자신을 포함한 가족 수를 나타내는 FamilySize 변수 생성, +1은 자기자신을 나타냄
    df['FamilySize'] = df["SibSp"] + df["Parch"] + 1

    # 혼자 탑승 여부 변수 생성
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Age*Class 변수 생성
    df['Age*Class'] = df.Age * df.Pclass

    # 요금대 변수 FareBand를 ordinal(0,1,2,3)하게 생성
    df['FareBand'] = df['Fare'].apply(lambda x: 0 if x <= 7.91 else (1 if x > 7.91 and x <= 14.454
                                                                 else (2 if x > 14.454 and x <= 31
                                                                       else 3)))
    return df

def drop_columns(df:pd.DataFrame) -> pd.DataFrame:
    """불필요한 Ticket, Cabin, Name, PassengerID 컬럼 제거"""
    df = df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
    return df

def split_dataframe(df:pd.DataFrame, train_num:int) -> pd.DataFrame:
    """합쳤던 데이터프레임을 분리"""
    train_df = df.iloc[:train_num].reset_index(drop=True)
    test_df = df.iloc[train_num:].reset_index(drop=True)
    return train_df, test_df

def preprocessing_pipeline(train_df:pd.DataFrame, test_df:pd.DataFrame, train_num:int) -> pd.DataFrame:
    df = integrate_dataframe(train_df, test_df)
    df = replace_null_feature(df)
    df = transform_feature_type(df)
    df = replace_null_age_median(df)
    df = make_derived_feature(df)
    df = drop_columns(df)
    tr_df, tt_df = split_dataframe(df, train_num)
    return tr_df, tt_df