import json
import pandas as pd

from flask import Flask, request, Response
from waitress import serve

app = Flask(__name__)

# Load Dataset
game_similarity_matrix = pd.read_csv('model_output/content_based_game_similarity_matrix.csv')
user_top_n_neighbours = json.load(open('model_output/collaborative_filtering_top_n_neighbours.json', 'r', encoding='utf-8'))

# Helper Data for Collaborative Filtering
reviews_df = pd.read_csv('final_data/user_recommendations_subset_clean.csv')
reviews_df = reviews_df[['app_id','user_id']]
games_df = pd.read_csv('final_data/game_details_subset_clean.csv')
game_ids = games_df['app_id'].tolist()
game_ids = [str(id) for id in game_ids]


# e.g. localhost:8888/api/ml/recommendations/game?id=1&size=10
@app.route('/api/ml/recommendations/game', methods=['GET'])
def get_content_based_game_similarity():
    query_app_id = request.args.get('id') or request.form.get('id')
    results_size = request.args.get('size') or request.form.get('size')
    
    result = game_similarity_matrix.loc[:,['app_id', query_app_id]].sort_values(by=query_app_id,ascending=False)
    result = result[result['app_id'] != int(query_app_id)]
    result_list = result['app_id'][0:int(results_size)].values.tolist()
    return Response(json.dumps(result_list), mimetype='application/json')


# e.g. localhost:8888/api/ml/recommendations/user?id=1&size=10
@app.route("/api/ml/recommendations/user", methods=['GET'])
def get_collaborative_filtering_user_recommendations():
    query_user_id = request.args.get('id') or request.form.get('id')
    results_size = request.args.get('size') or request.form.get('size')
    
    user_items = reviews_df[reviews_df['user_id'] == int(query_user_id)]['app_id'].tolist()
    user_items = [str(id) for id in user_items]
    
    if len(user_items) == 0:
        return []
    
    top_n_items = []
    for user_item in user_items:
        neighbor_list = user_top_n_neighbours.get(user_item, [])
        for neighbor in neighbor_list:
            if neighbor not in user_items and neighbor in game_ids:
                top_n_items.append(neighbor)
    
    top_n_items = list(set(top_n_items))[:int(results_size)]
    return Response(json.dumps(top_n_items), mimetype='application/json')


# run the server
if __name__ == '__main__':
    print("Starting the server.....")
    serve(app, host="0.0.0.0", port=8888)
