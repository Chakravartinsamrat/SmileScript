from flask import Flask, render_template, request, jsonify
from chatbot import get_response

app = Flask(__name__)

@app.route("/")
def index_get():
    return render_template("index.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #need to check if text is valid
    response = get_response(text)
    message={"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
