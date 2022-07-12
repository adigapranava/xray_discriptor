let WEB_URL = window.location.origin;


// loading symbol displayer
function loadingDisplay(params) {
    var formFilled = true;
    document.getElementById("img-form").querySelectorAll("[required]").forEach(function (i) {
        if (!i.value) {
            formFilled = false;
            return;
        };
    })
    if (formFilled) {
        document.getElementById("loader").innerHTML = "<div class=\"loader-div\"><div class=\"loader\"><div class=\"square\"></div><div class=\"path\"><div></div><div></div><div></div><div></div><div></div><div></div><div></div></div><p class=\"text-load\">Loading</p></div></div>";
    }
}