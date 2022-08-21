import cv2
import decomposition.algorithm as dec
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import utils.Helpers as Helpers
import segmentation.segmentation as seg
import nn.transforms as flowertransforms

MAIN_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static"
UPLOAD_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/upload"
SEGMENT_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/segment"
DECOMPOSE_FOLDER = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/decompose"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["MAIN_FOLDER"] = MAIN_FOLDER
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SEGMENT_FOLDER"] = SEGMENT_FOLDER
app.config["DECOMPOSE_FOLDER"] = DECOMPOSE_FOLDER
app.config["ORIGINAL_WIDTH"] = 0
app.config["ORIGINAL_HEIGHT"] = 0
app.config["ORIGINAL_FILENAME"] = ""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.form["upload_button"] == "Upload":
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
        return render_template("index.html")


@app.route("/upload/<filename>", methods=["GET", "POST"])
def upload_image(filename):
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
                full_filename = app.config["UPLOAD_FOLDER"] + "/" + filename  # os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(full_filename)
                return redirect(url_for("upload_image",
                                        filename=filename))
        elif request.form["submit_button"] == "Segment":
            # original_image = cv2.imread(app.config["UPLOAD_FOLDER"] + "/" + filename)
            # height, width = original_image.shape[:2]
            # app.config["ORIGINAL_WIDTH"] = width
            # app.config["ORIGINAL_HEIGHT"] = height
            segmap = segment(filename)
            app.config["ORIGINAL_FILENAME"] = filename
            segment_filename = Helpers.remove_file_extension(filename) + ".png"
            cv2.imwrite(app.config["SEGMENT_FOLDER"] + "/" +  segment_filename, segmap)
            return redirect(url_for("segment_view",
                                    filename=segment_filename))
    else:
        return render_template("upload_image.html", uploaded_image=filename)


@app.route("/segment/<filename>", methods=["GET", "POST"])
def segment_view(filename):
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
                full_filename = app.config["UPLOAD_FOLDER"] + "/" + filename  # os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(full_filename)
                return redirect(url_for("upload_image",
                                        filename=filename))
        elif request.form["submit_button"] == "Decompose":
            try:
                decompose(filename)
            except Exception as e:
                print(e)
                return render_template("segment.html", segmented_image=filename)
            return redirect(url_for("decompose_view", filename=filename))
    else:
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
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/display/segment/<filename>")
def display_segment_image(filename):
    return send_from_directory(app.config["SEGMENT_FOLDER"], filename)


@app.route("/display/decompose/<filename>")
def display_decompose_image(filename):
    return send_from_directory(app.config["DECOMPOSE_FOLDER"], filename)


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


def segment(filename):
    segmap = seg.segment_flower_parts(app.config["UPLOAD_FOLDER"] + "/" + filename)
    return segmap


def decompose(filename):
    segmap = cv2.imread(app.config["SEGMENT_FOLDER"] + "/" + filename)
    result, number_of_petals = dec.decomposition_algorithm(segmap)
    if number_of_petals == 0:
        raise Exception("Invalid segmentation, can't proceed with decomposition")
    # if numberof petals == 0 throw error
    original_filename = app.config["ORIGINAL_FILENAME"]
    original = cv2.imread(app.config["UPLOAD_FOLDER"] + "/" + original_filename)
    height, width = original.shape[:2]
    to_original_size = flowertransforms.RestoreOriginalSize((width, height))
    result = to_original_size(result)
    result = Helpers.separate_flower_parts(original, result, number_of_petals)
    cv2.imwrite(app.config["DECOMPOSE_FOLDER"] + "/" + filename, result)


@app.route("/shutdown", methods=["GET", "POST"])
def shutdown():
    shutdown_server()
    return "Server shutting down..."


app.run(host="127.0.0.1", port=8080, debug=True)
