from distutils.log import debug
from flask import Flask, render_template, redirect, request
import os
import helper as hlp

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = os.path.join(app.root_path, 'static/img/uploads')


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image1 = request.files["image1"]
            image2 = request.files["image2"]
            image1.save(os.path.join(app.root_path,
                                     'static/img/uploads', "img1.png"))
            image2.save(os.path.join(app.root_path,
                                     'static/img/uploads', "img2.png"))
            print(image1.filename)
            print(image2.filename)
            img1_path = os.path.join(app.root_path,
                                     'static/img/uploads', "img1")
            img2_path = os.path.join(app.root_path,
                                     'static/img/uploads', "img2")
            res = hlp.predict(image1, image2)
        
        return render_template("base.html", data={"report":res[0], "img1":img1_path, "img2":img2_path})

    return render_template("base.html")

if __name__ == '__main__':
    app.run(debug=True)
