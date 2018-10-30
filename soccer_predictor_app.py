import flask
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

#---------- MODEL IN MEMORY ----------------#

# Read the data on soccer match outcomes + team attributes,
# Build a Random Forest predictor on it


# load pickle files
path_folder = 'data/'
with open(path_folder + 'df_final.pkl','rb') as picklefile:
    df_final = pickle.load(picklefile)

# set target variable + features
y = df_final['winner']
X = df_final.iloc[:,1:]

# standardize X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# build Random Forest model
PREDICTOR = RandomForestClassifier(max_depth=7,n_estimators=16,class_weight='balanced').fit(X,y)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("index.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    
    # per results, create new dict for random forest model prediction
    data_updated = data["example"][0:4]
    
    if (data['example'][-2] == 0) & (data['example'][-1]==0):
        data_updated.extend([1.0,1.0,0.0,0.0])
    elif (data['example'][-2] == 0) & (data['example'][-1]==1):
        data_updated.extend([1.0,0.0,0.0,1.0])
    elif (data['example'][-2] == 1) & (data['example'][-1]==0):
        data_updated.extend([0.0,1.0,1.0,0.0])
    elif (data['example'][-2] == 1) & (data['example'][-1]==1):
        data_updated.extend([0.0,0.0,1.0,1.0])

    data["ex_updated"] = data_updated

    #data['example'] = [50.0,50.0,50.0,50.0,1.0,1.0,0.0,0.0]

    x = np.matrix(data["ex_updated"])
    score = PREDICTOR.predict(x)


    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)
