import cv2
import nn.learning as learning
import decomposition.algorithm as dec
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import utils.Helpers as Helpers

MAIN_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static"
UPLOAD_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/upload"
SEGMENT_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/segment"
DECOMPOSE_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/decompose"
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["MAIN_FOLDER"] = MAIN_FOLDER
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SEGMENT_FOLDER"] = SEGMENT_FOLDER
app.config["DECOMPOSE_FOLDER"] = DECOMPOSE_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    print("Method: ", request.method)
    if request.method == "POST":
        if request.form["upload_button"] == "Upload":

            print("request: ", request)
            print("Request files: ", request.files)
            if "file" not in request.files:
                print("No file part")
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                print("File not selected")
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                full_filename = app.config["UPLOAD_FOLDER"] + "/" + filename
                file.save(full_filename)
                print("Filename: ", full_filename)
                return redirect(url_for("upload_image",
                                        filename=filename))
    else:
        print("GET")
        return render_template("index.html")


@app.route("/upload/<filename>", methods=["GET", "POST"])
def upload_image(filename):
    if request.method == "POST":
        print("POST upload_image")
        if request.form["submit_button"] == "Upload":
            if "file" not in request.files:
                print("No file part")
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                print("File not selected")
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                full_filename = app.config["UPLOAD_FOLDER"] + "/" + filename  # os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(full_filename)
                return redirect(url_for("upload_image",
                                        filename=filename))
        elif request.form["submit_button"] == "Segment":
            print("Segment: ", filename)
            segmap = segment(filename)
            filename = Helpers.remove_file_extension(filename) + ".png"
            print("Filename no extension: ", filename)
            cv2.imwrite(app.config["SEGMENT_FOLDER"] + "/" + filename, segmap)
            return redirect(url_for("segment_view",
                                    filename=filename))
    else:
        print("GET upload image")
        return render_template("upload_image.html", uploaded_image=filename)


@app.route("/segment/<filename>", methods=["GET", "POST"])
def segment_view(filename):
    if request.method == "POST":
        print("POST segment_image")
        if request.form["submit_button"] == "Upload":
            if "file" not in request.files:
                print("No file part")
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                print("File not selected")
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                full_filename = app.config["UPLOAD_FOLDER"] + "/" + filename  # os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(full_filename)
                return redirect(url_for("upload_image",
                                        filename=filename))
        elif request.form["submit_button"] == "Decompose":
            print("Decompose: ", filename)
            filename_no_extension = Helpers.remove_file_extension(filename)
            print("Filename no extension: ", filename_no_extension)
            decompose(filename)
            return redirect(url_for("decompose_view", filename=filename))
    else:
        print("GET upload image")
        return render_template("segment.html", segmented_image=filename)


@app.route("/decompose/<filename>", methods=["GET", "POST"])
def decompose_view(filename):
    if request.method == "POST":
        if request.form["submit_button"] == "Upload":
            if "file" not in request.files:
                print("No file part")
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                print("File not selected")
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                full_filename = app.config["UPLOAD_FOLDER"] + "/" + filename
                file.save(full_filename)
                return redirect(url_for("upload_image",
                                        filename=filename))
    else:
        return render_template("decompose_view.html", decomposed_image=filename)


@app.route("/display/upload/<filename>")
def display_uploaded_image(filename):
    print("in display")
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/display/segment/<filename>")
def display_segment_image(filename):
    print("in display segment")
    return send_from_directory(app.config["SEGMENT_FOLDER"], filename)


@app.route("/display/decompose/<filename>")
def display_decompose_image(filename):
    print("in display decompose")
    return send_from_directory(app.config["DECOMPOSE_FOLDER"], filename)


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


def segment(filename):
    print("segmenting")
    segmap = learning.segment(app.config["UPLOAD_FOLDER"] + "/" + filename)
    return segmap


def decompose(filename):
    print("decomposing: ", filename)
    segmap = cv2.imread(app.config["SEGMENT_FOLDER"] + "/" + filename)
    result = dec.decomposition_algorithm(segmap)
    cv2.imwrite(app.config["DECOMPOSE_FOLDER"] + "/" + filename, result)


@app.route("/shutdown", methods=["GET", "POST"])
def shutdown():
    shutdown_server()
    return "Server shutting down..."


app.run(host="127.0.0.1", port=8080, debug=True)
