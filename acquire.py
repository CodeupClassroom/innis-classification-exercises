import os
import pandas as pd
import env

database_url_base = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/'

def get_titanic_data(use_cache=True):
    if os.path.exists('titanic.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('titanic.csv')
    print('Acquiring data from SQL database')
    query = 'SELECT * FROM passengers'
    df = pd.read_sql(query, database_url_base + 'titanic_db')
    df.to_csv('titanic.csv', index=False)
    return df

def get_iris_data(use_cache=True):
    if os.path.exists('iris.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('iris.csv')
    print('Acquiring data from SQL database')
    query = '''
    SELECT *
    FROM measurements
    JOIN species USING (species_id)
    '''
    df = pd.read_sql(query, database_url_base + 'iris_db')
    df.to_csv('iris.csv', index=False)
    return df

def get_telco_data(use_cache=True):
    if os.path.exists('telco.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('telco.csv')
    print('Acquiring data from SQL database')
    query = '''
    SELECT *
    FROM customers
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id)
    '''
    df = pd.read_sql(query, database_url_base + 'telco_churn')
    df.to_csv('telco.csv', index=False)
    return df

