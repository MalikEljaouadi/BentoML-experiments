import bentoml

from sklearn import svm, datasets

# Load dataset
iris = datasets.load_iris()

# Load model and train it
clf = svm.SVC(gamma='scale')
clf.fit(iris.data, iris. target)

# Save model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("irid_clf", clf)
print(f"Model saved: {saved_model}")