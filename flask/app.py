from flask import Flask, render_template, redirect, request
import os

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = os.path.join(app.root_path, 'static/img/uploads')


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image1 = request.files["image1"]
            image2 = request.files["image2"]
            image1.save(os.path.join(app.root_path,
                                     'static/img/uploads', image1.filename))
            image2.save(os.path.join(app.root_path,
                                     'static/img/uploads', image2.filename))
            print(image1.filename)
            print(image2.filename)
    return render_template("base.html")

if __name__ == '__main__':
    app.run()
