import os 
import pandas as pd
from src.constants.data_quality_check_constant import *
from src.entity.quality_check_artifact import *
from src.constants.data_transformation_constant import *
from src.entity.data_transformation_artifact import *

# constants
special_values = DataQualityConfig().special_values
max_threshold = DataQualityConfig().threshold

df_url_path = DataTransformationConfig().aggregated_artifact_interim_dir

path = os.path.join(df_url_path,'main_df_transformed.csv')

df = pd.read_csv(path)# special values


#downcast to reduce memory in ram
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')


#special values ratio dataframe
special_df = pd.DataFrame({
    'feature': df.columns,
    'special_ratio': df.isin(special_values).sum().values / df.shape[0]
})


#Zero values ratio dataframe
zero_df = (
    (df == 0).sum()
    .div(df.shape[0])
    .reset_index()
    .rename(columns={'index': 'feature', 0: 'zero_ratio'})
)

#Null values ratio dataframe
new_df = (df.isnull().sum() / df.shape[0]).to_frame('null_ratio').reset_index().rename(columns={'index': 'feature'})

#final feature quality dataframe
main_df_feature_quality_df = special_df.merge(zero_df,on='feature',how='left').merge(new_df,on='feature',how='left')
main_df_feature_quality_df

main_df_feature_quality_df['max_bad_ratio'] = (
    main_df_feature_quality_df[['special_ratio', 'zero_ratio', 'null_ratio']]
    .max(axis=1)
)


# flag the feature with higher max bad ratio
main_df_feature_quality_df['drop_feature'] = (
    main_df_feature_quality_df['max_bad_ratio'] > max_threshold
)

#make data quality dir inside the artifacts

artifact_dir = ARTIFACT_DIR
data_quality_dir = DataQualityConfig().data_quality_dir
data_quality_dir_path = os.path.join(artifact_dir,data_quality_dir)
os.makedirs(data_quality_dir_path,exist_ok=True)


# save the data quality check dataframe
feature_quality_csv_path = os.path.join(data_quality_dir_path,'main_df_feature_quality_df.csv')
main_df_feature_quality_df.to_csv(feature_quality_csv_path,index=False)

#check run of the file
print(f'feature_quality_csv: {feature_quality_csv_path}')
print(f'data_quality_dir_path:{data_quality_dir_path}')


def load_clean_df(input_csv:pd.DataFrame,feature_quality_csv_path:str):
    '''Drop bad features based on data quality report'''
    main_df_feature_quality_df = pd.read_csv(feature_quality_csv_path)
    
    temp = main_df_feature_quality_df[main_df_feature_quality_df['drop_feature']==True]
    drop_features_list = temp['feature'].values.tolist()
    
    input_csv = input_csv.drop(columns=drop_features_list)
    
    return input_csv

    