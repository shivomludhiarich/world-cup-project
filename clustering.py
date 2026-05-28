import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def get_player_info(player_name, player_stats, clustered_players):
    stats = player_stats[
        player_stats['player_name'] == player_name
    ].sort_values('season', ascending=False)
    
    cluster_info = clustered_players[
        clustered_players['player_name'] == player_name
    ]
    
    if len(stats) == 0:
        return None
    
    latest = stats.iloc[0]
    
    return {
        'name': player_name,
        'position': latest.get('position', 'Unknown'),
        'club': latest.get('current_club_name', 'Unknown'),
        'goals_per90': round(float(stats['goals_per90'].mean()), 3),
        'assists_per90': round(float(stats['assists_per90'].mean()), 3),
        'minutes_per_game': round(float(stats['minutes_per_game'].mean()), 1),
        'yellow_cards_per90': round(float(stats['yellow_cards_per90'].mean()), 3),
        'cluster_name': cluster_info['cluster_name'].values[0] if len(cluster_info) > 0 else 'Unknown'
    }

def get_similar_players(player_name, clustered_players, n=5):
    player = clustered_players[
        clustered_players['player_name'] == player_name
    ]
    
    if len(player) == 0:
        return []
    
    position = player['position'].values[0]
    
    same_position = clustered_players[
        (clustered_players['position'] == position) &
        (clustered_players['player_name'] != player_name)
    ].copy()
    
    features = ['pc1', 'pc2']
    
    player_coords = player[features].values
    all_coords = same_position[features].values
    
    distances = cdist(player_coords, all_coords, metric='euclidean')[0]
    same_position['distance'] = distances
    
    similar = same_position.nsmallest(n, 'distance')[
        ['player_name', 'cluster_name', 'goals_per90', 
         'assists_per90', 'minutes_per_game', 'distance']
    ]
    
    return similar.to_dict(orient='records')