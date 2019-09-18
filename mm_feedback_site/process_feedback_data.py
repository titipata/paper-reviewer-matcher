import os
import json
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    feedback_df = pd.read_json('data/ccn_feedback.json', orient='records', lines=True)
    feedback_df['timestamp'] = pd.to_datetime(feedback_df.timestamp, infer_datetime_format=True)
    feedback_df = feedback_df.sort_values('timestamp').groupby('registrant_id').last().reset_index().sort_values('timestamp')

    # text feedback from all responses
    n_text_feedback, n_responses = feedback_df.feedback_text.map(lambda x: x.strip() != '').sum(), len(feedback_df)
    print('number of response = {}, number of text feedback = {}, percentage = {} %'.format(n_responses, n_text_feedback, 100 * n_text_feedback / n_responses ))

    feedback_df['coi'] = feedback_df.coi.map(lambda x: ','.join(['1' if int(e) > 0 else '0' for e in x]))
    feedback_df['relevances'] = feedback_df.relevances.map(lambda x: ','.join(x))
    feedback_df['satisfactory'] = feedback_df.satisfactory.map(lambda x: ','.join(x))
    feedback_df.to_csv('data/ccn_2019_feedback.csv', index=False) # to send to organizer

    enjoyable = feedback_df.enjoyable.astype(int).values
    enjoyable = enjoyable[enjoyable > 0]
    print('average enjoyable score = {}'.format(enjoyable.mean()))

    usefulness = feedback_df.useful.astype(int).values
    usefulness = usefulness[usefulness > 0]
    print('average usefulness score = {}'.format(usefulness.mean()))