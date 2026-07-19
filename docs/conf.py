project = "femx"
author = "femx developers"
release = "0.3.0"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
]

root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
todo_include_todos = False

try:
    import sphinx_rtd_theme
except ImportError:
    html_theme = "classic"
    html_theme_options = {
        "codebgcolor": "lightgrey",
        "stickysidebar": "true",
    }
else:
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
