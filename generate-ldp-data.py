import os

import argparse
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split

DATA_DIR = Path("datasets/")

def byte_to_string_columns(data):
    for col, dtype in data.dtypes.items():
        if dtype == object:
            data[col] = data[col].apply(lambda x: x.decode("utf-8"))
    return data

def adult_dataset():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                   'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                   'native_country', 'label']
    
    def strip_string_columns(df):
        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

    dataset_train = pd.read_csv(DATA_DIR / 'income'/ 'adult.data', names=columns, na_values=['?', ' ?'])
    dataset_train = dataset_train.dropna(axis=0)
    dataset_train = dataset_train.drop(columns=['fnlwgt', 'education_num', 'occupation', 'native_country', 'relationship', 'capital_loss'])
    original_size = len(dataset_train)
    dataset_train["label"] = dataset_train["label"].str.replace(".","",regex=True)
    strip_string_columns(dataset_train)
    
    dataset_train['label'] = dataset_train['label'] == '>50K'
            
    dataset_train['race'] = dataset_train['race'] == 'White'
    dataset_train['workclass'] = dataset_train['workclass'] == 'Private'
    dataset_train['sex'] = dataset_train['sex'] == 'Male'
    dataset_train['marital_status'] = dataset_train.marital_status.str.contains('Married')
    dataset_train['education'] = dataset_train['education'].map({'10th': 'less-than-HS', '11th': 'less-than-HS', '9th': 'less-than-HS', 'Assoc-acdm': 'Assoc', 'Assoc-voc': 'Assoc',
                                                                 'Preschool': 'less-than-HS', 'Bachelors': 'College', 'Doctorate': 'College', 'HS-grad': 'HS', 'Masters': 'College',
                                                                'Prof-school': 'Prof-school', 'Some-college': 'HS'})
    # for one-hot encoding categorical variables
    cat_cols = dataset_train.select_dtypes(include="object")
    dataset_train = pd.get_dummies(dataset_train)

    # # for binarizing numerical variables
    num_cols = dataset_train.select_dtypes(include=['float64', 'int64'])
    for col in num_cols.columns:
        dataset_train[col] = np.where(dataset_train[col] >= dataset_train[col].mean(), 1, 0).astype('int')

    
    dataset_test = pd.read_csv(DATA_DIR / 'income' / 'adult.test', names=columns, na_values=['?', ' ?'])
    dataset_test = dataset_test.drop(columns=['fnlwgt', 'education_num', 'occupation', 'native_country', 'relationship', 'capital_loss'])
    strip_string_columns(dataset_test)

    dataset_test['label'] = dataset_test['label'] == '>50K.'
    dataset_test['race'] = dataset_test['race'] == 'White'
    dataset_test['workclass'] = dataset_test['workclass'] == 'Private'
    dataset_test['sex'] = dataset_test['sex'] == 'Male'
    dataset_test['marital_status'] = dataset_test.marital_status.str.contains('Married')
    dataset_test['education'] = dataset_test['education'].map({'10th': 'less-than-HS', '11th': 'less-than-HS', '9th': 'less-than-HS', 'Assoc-acdm': 'Assoc', 'Assoc-voc': 'Assoc',
                                                                 'Preschool': 'less-than-HS', 'Bachelors': 'College', 'Doctorate': 'College', 'HS-grad': 'HS', 'Masters': 'College',
                                                                'Prof-school': 'Prof-school', 'Some-college': 'HS'})
    # for one-hot encoding categorical variables
    cat_cols = dataset_test.select_dtypes(include="object")
    dataset_test = pd.get_dummies(dataset_test)

    # # for binarizing numerical variables
    num_cols = dataset_test.select_dtypes(include=['float64', 'int64'])
    for col in num_cols.columns:
        dataset_test[col] = np.where(dataset_test[col] >= dataset_test[col].mean(), 1, 0).astype('int')
    return dataset_train, dataset_test

def bank_dataset():
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
               'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    columns = {'V' + str(i + 1): v for i, v in enumerate(columns)}
    dataset = pd.DataFrame(loadarff(DATA_DIR / 'bank'/ 'phpkIxskf.arff')[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns=columns, inplace=True)
    dataset.rename(columns={'Class': 'label'}, inplace=True)
    
    dataset['label'] = dataset['label'] == '2'

    dataset = dataset.drop(columns=['job', 'marital', 'education', 'default', 'loan', 'contact', 'month'])
    dataset['housing'] = dataset['housing'] == 'yes'
    dataset['poutcome'] = dataset['poutcome'] == 'success'
    
    # for one-hot encoding categorical variables
    cat_cols = dataset.select_dtypes(include="object")
    dataset = pd.get_dummies(dataset)

    # # for binarizing numerical variables
    num_cols = dataset.select_dtypes(include='float64')
    for col in num_cols.columns:
        dataset[col] = np.where(dataset[col] >= dataset[col].mean(), 1, 0).astype('int')

    data_train, data_test = train_test_split(dataset, test_size=0.20)

    return data_train, data_test

def blood_dataset():
    columns = {'V1': 'recency', 'V2': 'frequency', 'V3': 'monetray', 'V4': 'time', 'Class': 'label'}
    dataset = pd.DataFrame(loadarff(DATA_DIR / 'blood'/ 'php0iVrYT.arff')[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns=columns, inplace=True)
   
    dataset['label'] = dataset['label'] == '2'

    # for one-hot encoding categorical variables
    cat_cols = dataset.select_dtypes(include="object")
    dataset = pd.get_dummies(dataset)

    # # for binarizing numerical variables
    num_cols = dataset.select_dtypes(include='float64')
    for col in num_cols.columns:
        dataset[col] = np.where(dataset[col] >= dataset[col].mean(), 1, 0).astype('int')

    data_train, data_test = train_test_split(dataset, test_size=0.20)

    return data_train, data_test

def calhousing_dataset():
    dataset = pd.DataFrame(loadarff(DATA_DIR / 'calhousing'/ 'houses.arff')[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
   
    
    # Make binary task by labelling upper half as true
    median_price = dataset['label'].median()
    dataset['label'] = dataset['label'] > median_price

    # for one-hot encoding categorical variables
    cat_cols = dataset.select_dtypes(include="object")
    dataset = pd.get_dummies(dataset)

    # # for binarizing numerical variables
    num_cols = dataset.select_dtypes(include='float64')
    for col in num_cols.columns:
        dataset[col] = np.where(dataset[col] >= dataset[col].mean(), 1, 0).astype('int')
   
    data_train, data_test = train_test_split(dataset, test_size=0.20)
    return data_train, data_test

def car_dataset():
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']
    dataset = pd.read_csv(DATA_DIR / 'car'/ 'car.data', names=columns)
    original_size = len(dataset)
    label_dict = {'unacc': 0, 'acc': 1, 'good': 1, 'vgood': 1}
    dataset['label'] = dataset['label'].replace(label_dict)

    dataset = dataset.drop(columns=['doors'])
    dataset['persons'] = dataset['persons'].map({'2': 0, '4': 1, 'more': 1})
    dataset['buying'] = dataset['buying'].map({'low': 'low', 'med': 'med', 'high': 'high', 'vhigh': 'high'})
    dataset['maint'] = dataset['maint'].map({'low': 'low', 'med': 'med', 'high': 'high', 'vhigh': 'high'})
    dataset['safety_dict'] = dataset['safety_dict'].map({'low': 'low', 'med': 'med', 'high': 'high', 'vhigh': 'high'})
    dataset['lug_boot'] = dataset['lug_boot'].map({'small': 0, 'med': 1, 'big': 1})
    
    # for one-hot encoding categorical variables
    cat_cols = dataset.select_dtypes(include="object")
    dataset = pd.get_dummies(dataset)

    # # for binarizing numerical variables
    num_cols = dataset.select_dtypes(include=['float64','int64'])
    for col in num_cols.columns:
        dataset[col] = np.where(dataset[col] >= dataset[col].mean(), 1, 0).astype('int')

    data_train, data_test = train_test_split(dataset, test_size=0.20)
    print(len(data_train), len(dataset.columns))
    return data_train, data_test

def diabetes_dataset():
    dataset = pd.read_csv(DATA_DIR / 'diabetes'/ 'diabetes.csv')
    original_size = len(dataset)
    dataset = dataset.rename(columns={'Outcome': 'label'})

    # for one-hot encoding categorical variables
    cat_cols = dataset.select_dtypes(include="object")
    dataset = pd.get_dummies(dataset)

    # # for binarizing numerical variables
    num_cols = dataset.select_dtypes(include=['int64', 'float64'])
    for col in num_cols.columns:
        dataset[col] = np.where(dataset[col] >= dataset[col].mean(), 1, 0).astype('int')

    data_train, data_test = train_test_split(dataset, test_size=0.20)
    print(len(data_train), len(dataset.columns))
    return data_train, data_test

def heart_dataset():
    dataset = pd.read_csv(DATA_DIR / 'heart'/ 'heart.csv')
    original_size = len(dataset)
    dataset = dataset.rename(columns={'HeartDisease': 'label'})
    dataset['ExerciseAngina'] = dataset['ExerciseAngina'].map({'N': 0, 'Y': 1})
    dataset['ChestPainType'] = dataset['ChestPainType'].map({'ASY': 0, 'ATA': 1, 'NAP': 0, 'TA':1 })
    dataset['RestingECG'] = dataset['RestingECG'] != 'Normal'
    dataset['Sex'] = dataset['Sex'].map({'F': 0, 'M': 1})
    dataset['ST_Slope'] = dataset['ST_Slope'] != 'Flat'
    
    # for one-hot encoding categorical variables
    cat_cols = dataset.select_dtypes(include="object")
    dataset = pd.get_dummies(dataset)

    # # for binarizing numerical variables
    num_cols = dataset.select_dtypes(include=['int64','float64'])
    for col in num_cols.columns:
        dataset[col] = np.where(dataset[col] >= dataset[col].mean(), 1, 0).astype('int')

    data_train, data_test = train_test_split(dataset, test_size=0.20)

    return data_train, data_test

def jungle_dataset():
    dataset = pd.DataFrame(loadarff(DATA_DIR / 'jungle'/ 'jungle_chess_2pcs_raw_endgame_complete.arff')[0])
    dataset = byte_to_string_columns(dataset)
    dataset.rename(columns={'class': 'label'}, inplace=True)
    dataset['label'] = dataset['label'] == 'w' 

    # for one-hot encoding categorical variables
    cat_cols = dataset.select_dtypes(include="object")
    dataset = pd.get_dummies(dataset)

    # # for binarizing numerical variables
    num_cols = dataset.select_dtypes(include='float64')
    for col in num_cols.columns:
        dataset[col] = np.where(dataset[col] >= dataset[col].mean(), 1, 0).astype('int')

    data_train, data_test = train_test_split(dataset, test_size=0.20)
    print(len(data_train), len(dataset.columns))
    return data_train, data_test

def GRR_Client(input_data, p):
    
    if np.random.binomial(1, p) == 1:
        return int(input_data)

    else:
        return int(1 - input_data)

def gen_keys(num_feat):
    total = 2 ** (num_feat)
    possible_keys = ['0' for _ in range(int(total/2))]
    possible_keys.extend(['1' for _ in range(int(total/2))])
    
    rounds = [i for i in range(num_feat)]
   
    for r in rounds[::-1]:
        if r == 0:
            continue

        count = 0
        for i, k in enumerate(possible_keys):
            if count < 2**(r-1):
                possible_keys[i] = k + '0'
            else:
                possible_keys[i] = k +'1'
            count += 1
            if count == 2**r:
                count = 0
                
    return possible_keys

def generate_data(epsilon, dfTrain):

    # make sure label is last column
    train_labels = dfTrain.pop('label')
    dfTrain.insert(len(dfTrain.columns), 'label', train_labels)

    assert dfTrain.columns[-1] == 'label'
    
    # Do randomized response on TRAIN ONLY
    if epsilon:
        epsilon = epsilon / len(dfTrain.columns)
        p = np.exp(epsilon) / (np.exp(epsilon) + 1)
 
        ldp_dftrain = []

        for col in dfTrain.columns:
            df_new_col = pd.DataFrame([int(GRR_Client(val, p)) for val in dfTrain[col]], columns=[col])
            ldp_dftrain.append(df_new_col)
   
        dfTrain_rr = pd.concat(ldp_dftrain, axis=1)

        # perform reconstruction
        num_repeat = len(dfTrain_rr.columns)

        possible_keys =  gen_keys(num_repeat)
 
        lambda_dict = {}
       
        for key in possible_keys:
            lambda_dict[key] = 0
    
        for index, row in dfTrain_rr.iterrows():
            key = ''.join(str(x) for x in row)
            lambda_dict[key] += 1
        
        p_ = np.linalg.inv([[p, 1-p],[1-p, p]])

        # get P^-1
        for n in range(num_repeat):
            if n == 0:
                continue
            else:
                b = np.linalg.inv([[p, 1-p], [1-p, p]])
                p_ = np.kron(p_, b)
            
        # construct big lambda in order
        keys = list(lambda_dict.keys())
        keys.sort()
        sorted_lambda_dict = {i: lambda_dict[i] for i in keys}
        
        lambda_list = [lambda_dict[k]/len(dfTrain) for k in keys] #lambda hat 

        pi_tilde = np.matmul(p_, lambda_list)
       
        for i, pi in enumerate(pi_tilde):
            if pi < 0:
                pi_tilde[i] = 0

        pi_tilde_scaled = np.true_divide(pi_tilde, np.sum(pi_tilde))
       
        pi_tilde_list = [np.ceil(pi*len(dfTrain)) for pi in pi_tilde_scaled]
      
        recon_train = []

        for i, counts in enumerate(pi_tilde_list):
            for j in range(int(counts)):
                recon_train.append([int(elem) for elem in keys[i]])
        
        recon_train = pd.DataFrame(recon_train, columns=dfTrain.columns)
        recon_train = recon_train.sample(len(dfTrain)).reset_index(drop=True)
        
        return recon_train
    else:
        return dfTrain

def call_dataset(dataset):
    if dataset == 'adult':
        return adult_dataset()
    elif dataset == 'bank':
        return bank_dataset()
    elif dataset == 'blood':
        return blood_dataset()
    elif dataset == 'calhousing':
        return calhousing_dataset()
    elif dataset == 'car':
        return car_dataset()
    elif dataset == 'diabetes':
        return diabetes_dataset()
    elif dataset == 'jungle':
        return jungle_dataset()
    else:
        print('Dataset not supported. Please select from: [adult, bank, blood, calhousing, car, diabetes, jungle')
        exit()

def main(dataset, epsilon, rounds):
    dataset_name = dataset
    dfTrain, dfTest = call_dataset(dataset)

    Path(f'ldp-non-serial/').mkdir(parents=True, exist_ok=True)
    for r in range(rounds):
        full_data = pd.concat([dfTrain, dfTest], axis=0, ignore_index=True)
        dfTest_ = full_data.sample(frac=.2, replace=False)
        dfTrain_ = full_data.drop(dfTest_.index, axis=0)
    
        if epsilon:
            dfTest_.to_csv(f'ldp-non-serial/{dataset_name}_ldp_eps_{epsilon}_test_{r+1}.csv', encoding='utf-8', index=False)
            recon_train = generate_data(epsilon, dfTrain_)
            recon_train.to_csv(f'ldp-non-serial/{dataset_name}_ldp_eps_{epsilon}_train_{r+1}.csv', encoding='utf-8', index=False)
        else:
            dfTest_.to_csv(f'ldp-non-serial/{dataset_name}_nodp_test_{r+1}.csv', encoding='utf-8', index=False)
            recon_train = generate_data(epsilon, dfTrain_)
            recon_train.to_csv(f'ldp-non-serial/{dataset_name}_nodp_train_{r+1}.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', dest="dataset", help="Choice from: adult, bank, blood, calhousing, car, diabetes, heart, jungle.", type=str)
    parser.add_argument('--e', dest="epsilon", help="Choice of epsilon. Takes one int value or None", type=int)
    parser.add_argument('--r', dest="rounds", help="Number of training files to generate", type=int)
    args = parser.parse_args()
    main(args.dataset, args.epsilon, args.rounds)