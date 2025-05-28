

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data

df = pd.read_csv("heart_disease_dataset.csv")

#Create a training matrix without the target variable (Heart Diseas)

target_column = df.columns[-1]
df_features = df.drop(columns = [target_column])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features

num_features = df_features.shape[1]

removed_feature_after_iter = []
pc1 = []

# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature

    reduced_data = np.delete(scaled_data, i, axis = 1)
    
    

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)
    
    

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that

    removed_feature_after_iter.append(df_features.columns[i])
    pc1.append(pca.explained_variance_ratio_[0])


# Find the maximum PC1 variance


max_pc1 = max(pc1)
pc1_arr = np.array(pc1)
max_pc1_var = np.argmax(pc1)
removed_feat_max = removed_feature_after_iter[max_pc1_var]

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {max_pc1} when removing {removed_feat_max}")




