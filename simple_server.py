import json
import pandas as pd

from flask import Flask, request, Response
from waitress import serve

app = Flask(__name__)
cs_matrix = pd.read_csv('final_data/content-based-similarity-matrix.csv')

# e.g. localhost:8080/recommendations?id=550&start=0&end=10
@app.route('/api/recommendations/', methods=['GET'])
def get_similar_list():
    query_app_id = request.args.get('id') or request.form.get('id')
    index_start = request.args.get('start') or request.form.get('start')
    index_end = request.args.get('end') or request.form.get('end')
    
    result = cs_matrix.loc[:,['app_id', query_app_id]].sort_values(by=query_app_id,ascending=False)
    result = result[result['app_id'] != int(query_app_id)]
    result_list = result['app_id'][int(index_start):int(index_end)].values.tolist()
    return Response(json.dumps(result_list),  mimetype='application/json')

# run the server
if __name__ == '__main__':
    print("Starting the server.....")
    serve(app, host="0.0.0.0", port=8080)
