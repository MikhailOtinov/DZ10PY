import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

lst = ["robot"] * 10 + ["human"] * 10
random.shuffle(lst)

data = pd.DataFrame({"whoAmI'": lst})

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(np.array(data["whoAmI'"]).reshape(-1, 1))

unique_values = data["whoAmI'"].unique()

column_names = [f"whoAmI'_{value}" for value in unique_values]

data_encoded = pd.DataFrame(one_hot_encoded, columns=column_names)

data_encoded = pd.concat([data, data_encoded], axis=1)

data_encoded.drop(columns=["whoAmI'"], inplace=True)

print(data_encoded.head())