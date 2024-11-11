import sys, os
from ultralytics import YOLO
from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.exception import SignException
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!"


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predictRoute():
    try:
        image = request.json["image"]
        decodeImage(image, clApp.filename)
        os.system(
            "yolo detect predict model='best.pt' conf=0.25 source='inputImage.jpg' save=True"
        )
        opencodedbase64 = encodeImageIntoBase64("runs/detect/predict/inputImage.jpg")
        result = {"image": opencodedbase64.decode("utf-8")}
        os.system("rm -rf runs/detect/predict")

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


@app.route("/live", methods=["GET"])
@cross_origin()
def predictLive():
    try:
        os.system("yolo detect predict model='best.pt' imgsz=640 source=0 show=True")
        os.system("rm -rf runs/detect/predict")
        return "Camera starting!!"

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)
