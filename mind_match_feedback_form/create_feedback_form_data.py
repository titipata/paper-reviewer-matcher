"""

"""
import pandas as pd
import json

if __name__ == '__main__':
    # read matched tables, full dataset
    match_df = pd.read_csv('../CCN_2019/ccn_mindmatch_2019.csv')
    df = pd.read_csv('../CCN_2019/CN19_MindMatchData_20190903-A.csv')
    df = df.merge(match_df, how='left', on='RegistrantID')
    for i in range(0, 6):
        df['table_%s' % str(i)] = df.ScheduleTables.map(lambda x: x.split('|')[i])

    information = {}
    for _, r in df.iterrows():
        information[r['RegistrantID']] = {
            'registrant_id': r['RegistrantID'],
            'full_name': r['full_name'], 
            'email': r['Email'],
            'affiliation': r['Affiliation']
        }

    mind_match_forms = []
    for _, r in df.iterrows():
        tables = r['ScheduleTables'].split('|')
        matches = []
        for i, table in enumerate(tables):
            mind_match_id = list(set(df[df['table_%s' % i] == table].RegistrantID.values) - {r['RegistrantID']})[0]
            matches.append(information[mind_match_id])
        mind_match_forms.append({
            'registrant_id': r['RegistrantID'],
            'full_name': r['full_name'], 
            'email': r['Email'],
            'affiliation': r['Affiliation'],
            'matches_info': matches
        })
    json.dump(mind_match_forms, open('../CCN_2019/ccn_mind_match_feedback_form.json', 'w'), indent=4)