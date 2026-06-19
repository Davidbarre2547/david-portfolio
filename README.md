# David Projects

Static GitHub Pages site for the Python projects in this workspace.

## Publish with GitHub Pages

1. Push this folder to a GitHub repository.
2. In the repository settings, open **Pages**.
3. Set the source to the `main` branch and the root folder.
4. Save and wait for the site to build.

## Files shown on the site

- [Emotion Detection.py](Emotion%20Detection.py)
- [Face Filter.py](Face%20Filter.py)
- [Gemini ChatBot.py](Gemini%20ChatBot.py)
- [Youtube Summarizer.py](Youtube%20Summarizer.py)

## Running the projects

Install dependencies first:

```bash
pip install -r requirements.txt
```

- **Emotion Detection** – put images in the `target` folder (defaults to `/content/` for Colab) and run `python "Emotion Detection.py"`.
- **Face Filter** – needs a webcam and a `sunglasses.png` overlay in the folder, then `python "Face Filter.py"` (press `q` to quit).
- **Gemini ChatBot** – set a Gemini key via the `DAVID_GEMINI_API_KEY` Colab secret or the `GEMINI_API_KEY` environment variable, then run the script (launches a Gradio UI).
- **Youtube Summarizer** – set `GEMINI_API_KEY` in a `.env` file, then `streamlit run "Youtube Summarizer.py"`.