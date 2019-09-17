import os
import json
import flask
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from datetime import timezone, datetime


def read_json(file_path):
    """
    Read collected file from path
    """
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as fp:
            ls = [json.loads(line) for line in fp]
        return ls


def save_json(ls, file_path):
    """
    Save list of dictionary to JSON
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in ls))


app = Flask(__name__,
            template_folder='flask_templates')
app.secret_key = 'made at Kording Lab.'
app.config['TEMPLATES_AUTO_RELOAD'] = True

MINDMATCH_DATA_PATH = 'data/ccn_mind_match_feedback_form.json'
FEEDBACK_DATA_PATH = 'data/ccn_feedback.json'
mind_match_data = json.load(open(MINDMATCH_DATA_PATH, 'r'))


@app.route("/", methods=['GET', 'POST'])
def index():
    return flask.render_template('index.html')


@app.route('/regid/<reg_id>')
def feedback_form(reg_id):
    try:
        data = [d for d in mind_match_data
                if d['registrant_id'] == int(reg_id)][0]
        for match in data['matches_info']:
            match.pop('registrant_id', None)
        data.update({
            'enumerate': enumerate,
        })
    except:
        data = {
            "registrant_id": 0,
            "full_name": "John Doe",
            "email": "john_doe@gmail.com",
            "affiliation": "Random University",
            "matches_info": [],
            "enumerate": enumerate
        }
    return flask.render_template('feedback.html', **data)


@app.route('/handle_submit/', methods=['GET', 'POST'])
def handle_submit():
    # save data here
    if request.method == 'POST':
        feedback_data = read_json(FEEDBACK_DATA_PATH)
        registrant_id = request.form['registrant_id']
        feedback_text = request.form.get('text_input', '')
        relevances = [request.form.get('relevance_%s' % i, '0')
                      for i in range(0, 6)]
        satisfactory = [request.form.get(
            'satisfactory_%s' % i, '0') for i in range(0, 6)]
        coi = [request.form.get('coi_%s' % i, '0') for i in range(0, 6)]
        arrange_before = request.form.get('before_checkbox', '0')
        feedback_data.append({
            'registrant_id': registrant_id,
            'relevances': relevances,
            'satisfactory': satisfactory,
            'coi': coi,
            'feedback_text': feedback_text,
            'arrange_before': arrange_before
        })
        save_json(feedback_data, FEEDBACK_DATA_PATH)
    # return to default page
    return flask.redirect('/')


if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
