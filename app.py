from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import json
import pandas as pd
import numpy as np
from predictor import predict_match
from clustering import get_player_info, get_similar_players
from predictor import predict_score


# add this dictionary to app.py at the top after imports
COUNTRY_CODES = {
    'Afghanistan': 'af', 'Albania': 'al', 'Algeria': 'dz', 'Angola': 'ao',
    'Argentina': 'ar', 'Armenia': 'am', 'Australia': 'au', 'Austria': 'at',
    'Azerbaijan': 'az', 'Bahrain': 'bh', 'Bangladesh': 'bd', 'Belarus': 'by',
    'Belgium': 'be', 'Bolivia': 'bo', 'Bosnia and Herzegovina': 'ba',
    'Brazil': 'br', 'Bulgaria': 'bg', 'Burkina Faso': 'bf', 'Cameroon': 'cm',
    'Canada': 'ca', 'Chile': 'cl', 'China': 'cn', 'Colombia': 'co',
    'Congo': 'cg', 'Costa Rica': 'cr', 'Croatia': 'hr', 'Cuba': 'cu',
    'Czech Republic': 'cz', 'Denmark': 'dk', 'Ecuador': 'ec', 'Egypt': 'eg',
    'England': 'gb-eng', 'Estonia': 'ee', 'Ethiopia': 'et', 'Finland': 'fi',
    'France': 'fr', 'Gabon': 'ga', 'Georgia': 'ge', 'Germany': 'de',
    'Ghana': 'gh', 'Greece': 'gr', 'Guatemala': 'gt', 'Guinea': 'gn',
    'Honduras': 'hn', 'Hungary': 'hu', 'Iceland': 'is', 'India': 'in',
    'Indonesia': 'id', 'Iran': 'ir', 'Iraq': 'iq', 'Ireland': 'ie',
    'Israel': 'il', 'Italy': 'it', 'Ivory Coast': 'ci', 'Jamaica': 'jm',
    'Japan': 'jp', 'Jordan': 'jo', 'Kazakhstan': 'kz', 'Kenya': 'ke',
    'Kosovo': 'xk', 'Kuwait': 'kw', 'Kyrgyzstan': 'kg', 'Latvia': 'lv',
    'Lebanon': 'lb', 'Libya': 'ly', 'Lithuania': 'lt', 'Luxembourg': 'lu',
    'Mali': 'ml', 'Malta': 'mt', 'Mexico': 'mx', 'Moldova': 'md',
    'Montenegro': 'me', 'Morocco': 'ma', 'Mozambique': 'mz', 'Netherlands': 'nl',
    'New Zealand': 'nz', 'Nicaragua': 'ni', 'Nigeria': 'ng',
    'North Korea': 'kp', 'North Macedonia': 'mk', 'Norway': 'no', 'Oman': 'om',
    'Pakistan': 'pk', 'Palestine': 'ps', 'Panama': 'pa', 'Paraguay': 'py',
    'Peru': 'pe', 'Philippines': 'ph', 'Poland': 'pl', 'Portugal': 'pt',
    'Qatar': 'qa', 'Romania': 'ro', 'Russia': 'ru', 'Rwanda': 'rw',
    'Saudi Arabia': 'sa', 'Scotland': 'gb-sct', 'Senegal': 'sn', 'Serbia': 'rs',
    'Slovakia': 'sk', 'Slovenia': 'si', 'Somalia': 'so', 'South Africa': 'za',
    'South Korea': 'kr', 'Spain': 'es', 'Sudan': 'sd', 'Sweden': 'se',
    'Switzerland': 'ch', 'Syria': 'sy', 'Taiwan': 'tw', 'Tajikistan': 'tj',
    'Tanzania': 'tz', 'Thailand': 'th', 'Togo': 'tg', 'Trinidad and Tobago': 'tt',
    'Tunisia': 'tn', 'Turkey': 'tr', 'Turkmenistan': 'tm', 'Uganda': 'ug',
    'Ukraine': 'ua', 'United Arab Emirates': 'ae', 'United States': 'us',
    'Uruguay': 'uy', 'Uzbekistan': 'uz', 'Venezuela': 've', 'Vietnam': 'vn',
    'Wales': 'gb-wls', 'Yemen': 'ye', 'Zambia': 'zm', 'Zimbabwe': 'zw',
    'USA': 'us', 'South Sudan': 'ss', 'DR Congo': 'cd', 'Cape Verde': 'cv',
    'Equatorial Guinea': 'gq', 'Eswatini': 'sz', 'Faroe Islands': 'fo',
    'Finland': 'fi', 'Haiti': 'ht', 'Kosovo': 'xk', 'Liberia': 'lr',
    'Malawi': 'mw', 'Mauritania': 'mr', 'Namibia': 'na', 'Niger': 'ne',
    'Sierra Leone': 'sl', 'Cyprus': 'cy', 'El Salvador': 'sv',
    'Liechtenstein': 'li', 'Andorra': 'ad', 'San Marino': 'sm',
    'Northern Ireland': 'gb-nir', 'New Caledonia': 'nc',
    'Central African Republic': 'cf', 'Benin': 'bj', 'Burundi': 'bi',
    'Comoros': 'km', 'Djibouti': 'dj', 'Eritrea': 'er', 'Gambia': 'gm',
    'Guinea-Bissau': 'gw', 'Lesotho': 'ls', 'Madagascar': 'mg',
    'Mauritius': 'mu', 'São Tomé and Príncipe': 'st', 'Seychelles': 'sc',
    'Chad': 'td', 'Congo DR': 'cd', 'Côte d\'Ivoire': 'ci'
}




app = Flask(__name__)

# ── load all models and data on startup ──────────────
print("Loading models...")

# Module A
rf_model = joblib.load('models/match_predictor_model.pkl')

with open('models/elo_ratings.pkl', 'rb') as f:
    final_elo = pickle.load(f)

with open('models/feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

# Module B
kmeans_att = joblib.load('models/kmeans_attackers.pkl')
kmeans_mid = joblib.load('models/kmeans_midfielders.pkl')
kmeans_def = joblib.load('models/kmeans_defenders.pkl')
scaler_att = joblib.load('models/scaler_attackers.pkl')
scaler_mid = joblib.load('models/scaler_midfielders.pkl')
scaler_def = joblib.load('models/scaler_defenders.pkl')

# load data
df_matches = pd.read_csv('data/matches_full.csv')
df_matches['date'] = pd.to_datetime(df_matches['date'])
clustered_players = pd.read_csv('data/clustered_players.csv')
player_stats = pd.read_csv('data/player_stats.csv')
value_performance = pd.read_csv('data/value_performance.csv')
valuations = pd.read_csv('data/valuations_filtered.csv')
valuations['date'] = pd.to_datetime(valuations['date'])

print("All models and data loaded!")

# ── routes ────────────────────────────────────────────
@app.route('/api/country_codes', methods=['GET'])
def api_country_code():
    country = request.args.get('country', '')
    if country:
        code = COUNTRY_CODES.get(country, '').lower()
        return jsonify({'code': code})
    return jsonify({k: v.lower() for k, v in COUNTRY_CODES.items()})

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predictor')
def predictor():
    # get all unique teams for dropdowns
    teams = sorted(list(set(
        df_matches['home_team'].tolist() + 
        df_matches['away_team'].tolist()
    )))
    return render_template('predictor.html', teams=teams, country_codes=COUNTRY_CODES)


@app.route('/players')
def players():
    return render_template('players.html')

@app.route('/clusters')
def clusters():
    return render_template('clusters.html')

# ── API endpoints ─────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    neutral = data.get('neutral', False)
    
    # step 1 — RF win probability
    result = predict_match(
        home_team, away_team, neutral,
        rf_model, final_elo, df_matches, feature_cols
    )
    
    # step 2 — Poisson score using RF probabilities
    score = predict_score(
        home_team, away_team, neutral,
        final_elo, df_matches,
        result['Home win'],   # pass RF probs directly
        result['Away win']
    )
    
    # merge both into one response
    result['home_goals'] = score['home_goals']
    result['away_goals'] = score['away_goals']
    result['home_xg'] = score['home_xg']
    result['away_xg'] = score['away_xg']
    
    return jsonify(result)

@app.route('/api/h2h', methods=['POST'])
def api_h2h():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    
    # get all matches between these two teams
    h2h = df_matches[
        ((df_matches['home_team'] == home_team) & (df_matches['away_team'] == away_team)) |
        ((df_matches['home_team'] == away_team) & (df_matches['away_team'] == home_team))
    ].sort_values('date', ascending=False).head(5)
    
    if len(h2h) == 0:
        return jsonify({'message': 'no record found'})
    
    results = []
    for _, row in h2h.iterrows():
        # determine result from perspective of home_team
        if row['home_team'] == home_team:
            if row['home_score'] > row['away_score']:
                result = 'W'
            elif row['home_score'] < row['away_score']:
                result = 'L'
            else:
                result = 'D'
            score = f"{int(row['home_score'])} - {int(row['away_score'])}"
        else:
            if row['away_score'] > row['home_score']:
                result = 'W'
            elif row['away_score'] < row['home_score']:
                result = 'L'
            else:
                result = 'D'
            score = f"{int(row['away_score'])} - {int(row['home_score'])}"
        
        results.append({
            'date': pd.to_datetime(row['date']).strftime('%d %b %Y'),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': int(row['home_score']),
            'away_score': int(row['away_score']),
            'score': score,
            'result': result,
            'tournament': row['tournament']
        })
    
    return jsonify(results)

@app.route('/api/player_search', methods=['GET'])
def api_player_search():
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])
    
    matches = player_stats[
        player_stats['player_name'].str.contains(query, case=False, na=False)
    ]['player_name'].unique().tolist()[:10]
    
    return jsonify(matches)

@app.route('/api/player_stats', methods=['GET'])
def api_player_stats():
    player_name = request.args.get('name', '')
    
    stats = player_stats[
        player_stats['player_name'] == player_name
    ].sort_values('season', ascending=False).head(5)
    
    if len(stats) == 0:
        return jsonify({'error': 'Player not found'})
    
    records = stats.to_dict(orient='records')
    player_cluster = clustered_players[clustered_players['player_name'] == player_name]
    cluster_name = player_cluster.iloc[0]['cluster_name'] if len(player_cluster) > 0 else 'Unknown'
    
    for r in records:
        r['cluster_name'] = cluster_name
        
    return jsonify(records)

@app.route('/api/clusters_data', methods=['GET'])
def api_clusters_data():
    position = request.args.get('position', 'Attack')
    
    data = clustered_players[
        clustered_players['position'] == position
    ][['player_name', 'pc1', 'pc2', 'cluster_name', 
       'goals_per90', 'assists_per90', 'minutes_per_game']].dropna()
    
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/market_value', methods=['GET'])
def api_market_value():
    player_name = request.args.get('name', '')
    
    player_id_rows = player_stats[
        player_stats['player_name'] == player_name
    ]['player_id']
    
    if len(player_id_rows) == 0:
        return jsonify({'error': 'Player not found'})
    
    player_id = player_id_rows.values[0]
    
    val_history = valuations[
        valuations['player_id'] == player_id
    ].sort_values('date')[['date', 'market_value_in_eur']].copy()
    
    val_history['date'] = val_history['date'].dt.strftime('%Y-%m-%d')
    val_history['market_value_in_eur'] = (
        val_history['market_value_in_eur'] / 1e6
    ).round(1)
    
    return jsonify(val_history.to_dict(orient='records'))

@app.route('/api/similar_players', methods=['GET'])
def api_similar_players():
    player_name = request.args.get('name', '')
    from clustering import get_similar_players
    result = get_similar_players(player_name, clustered_players)
    return jsonify(result)

@app.route('/api/top_players', methods=['GET'])
def api_top_players():
    position = request.args.get('position', 'Attack')
    metric = request.args.get('metric', 'goals_per90')
    
    top = value_performance[
        value_performance['position'] == position
    ].nlargest(20, metric)[
        ['name', metric, 'value_millions', 'cluster_name']
    ]
    
    return jsonify(top.to_dict(orient='records'))


@app.route('/debug/teams')
def debug_teams():
    teams = sorted(list(set(
        df_matches['home_team'].tolist() + 
        df_matches['away_team'].tolist()
    )))
    # check which teams have country codes
    missing = [t for t in teams if t not in COUNTRY_CODES]
    matched = [t for t in teams if t in COUNTRY_CODES]
    return jsonify({'matched': matched[:10], 'missing': missing[:20]})

if __name__ == '__main__':
    app.run(debug=True)
