# Machine learning and musical layout

To compile this presentation, make sure you have a sane distribution of python 3.x. Then, do something like:

```bash
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ landslide -x tables slurs.md
```

This will output a file called `presentation.html` that is the presentation. You can also access this on [netlify](https://ml-music-notation.netlify.com). Enjoy!