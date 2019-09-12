from rulelearning.imli import imli
import pandas as pd

data = pd.read_csv("iris_bintarget.csv", sep=",", header=0, error_bad_lines=False)
print(data)

model = imli()

X, y = model.discretize("iris_bintarget.csv")
print(X)
print(len(X[0]))
print(y)
print(len(y))
