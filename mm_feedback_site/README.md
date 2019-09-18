# Feedback form for CCN 2019

Create a form to get feedback from Mind-Matching event at CCN 2019.

First, put data in `data` folder naming e.g. `ccn_mind_match_feedback_form.json`. 

Then, edit `main.py` for `MINDMATCH_DATA_PATH` (path for data e.g. `ccn_mind_match_feedback_form.json`) and `FEEDBACK_DATA_PATH` (path to save responses)


```bash
export FLASK_APP=main.py
flask run --host=0.0.0.0 --port=5555
```

