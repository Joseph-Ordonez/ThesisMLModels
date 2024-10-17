import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Create App
app = Flask(__name__)

#Pickle model load
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)

        return jsonify({
                "prediction": prediction[0]
            })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

    # return render_template("index.html", prediction_text = "The recommendation is: {}".format(prediction))


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Ver el JSON recibido
#         print(request.get_json(force=True))
#         data = request.get_json(force=True)
#         float_features = [float(x) for x in data["features"]]
        
#         if len(float_features) != 87:
#             return jsonify({"error": "Se requieren 87 características."}), 400
        
#         features = [np.array(float_features)]
#         prediction = model.predict(features)
        
#         return jsonify({
#             "prediction": prediction[0]
#         })
    
#     except Exception as e:
#         return jsonify({
#             "error": str(e)
#         }), 400


if __name__ == "__main__":
    app.run(debug=True)
