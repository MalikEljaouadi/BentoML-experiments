import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# Get the saved model (in the model_packaging.py) from the BentoML local model store
iris_clf_runner =bentoml.sklearn.get("irid_clf:dx4rabxfvwocnkrb").to_runner()

# Creating the service (a Service is the core componenet of BentoML that will allow us to expose the model to an API)
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

# Expose the service in an API
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray)-> np.ndarray:
    return iris_clf_runner.predict.run(input_series)

# To lunch the Service exposing this API we use the following command: bentoml serve service:svc --reload
# This command will lunch the server and expose its swagger file on http://0.0.0.0:3000/
