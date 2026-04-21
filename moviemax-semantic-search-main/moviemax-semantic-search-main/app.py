from flask import Flask, render_template, request, abort
from catalog_service import get_movie_detail
from search_service import get_page_number, search_movies

app = Flask(__name__)


@app.route("/")
def home():
    """Render the landing page."""
    return render_template("index.html")


@app.route("/search")
def search():
 
    query = request.args.get("q", "").strip()
    current_page = get_page_number(request)

    search_result = search_movies(query=query, current_page=current_page)

    return render_template(
        "results.html",
        query=query,
        current_page=current_page,
        **search_result,
    )


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
  
    movie = get_movie_detail(movie_id)

    if movie is None:
        abort(404)

    return render_template("movie.html", movie=movie)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
