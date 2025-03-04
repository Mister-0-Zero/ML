from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False для массива NumPy
data = [["Red"], ["Blue"], ["Green"], ["Blue"], ["Red"]]

encoded = encoder.fit_transform(data)
print(encoded)
