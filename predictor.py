import pandas as pd
import numpy as np

def get_recent_form(team, date, df, n=10):
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].sort_values('date').tail(n)
    
    if len(team_matches) < 5:
        return 0.5
    
    wins = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team and row['outcome'] == 'Home win':
            wins += 1
        elif row['away_team'] == team and row['outcome'] == 'Away win':
            wins += 1
    
    return round(wins / len(team_matches), 3)

def get_h2h_record(home_team, away_team, date, df, n=10):
    h2h_matches = df[
        (
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ) &
        (df['date'] < date)
    ].sort_values('date').tail(n)
    
    if len(h2h_matches) < 2:
        return 0.5
    
    home_wins = 0
    for _, row in h2h_matches.iterrows():
        if row['home_team'] == home_team and row['outcome'] == 'Home win':
            home_wins += 1
        elif row['away_team'] == home_team and row['outcome'] == 'Away win':
            home_wins += 1
    
    return round(home_wins / len(h2h_matches), 3)

def get_h2h_draw_rate(home_team, away_team, date, df, n=10):
    h2h_matches = df[
        (
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ) &
        (df['date'] < date)
    ].sort_values('date').tail(n)
    
    if len(h2h_matches) < 2:
        return 0.25
    
    draws = len(h2h_matches[h2h_matches['outcome'] == 'Draw'])
    return round(draws / len(h2h_matches), 3)

def get_goal_averages(team, date, df, n=10):
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].sort_values('date').tail(n)
    
    if len(team_matches) < 5:
        return 1.2, 1.2
    
    goals_scored = 0
    goals_conceded = 0
    
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            goals_scored += row['home_score']
            goals_conceded += row['away_score']
        else:
            goals_scored += row['away_score']
            goals_conceded += row['home_score']
    
    return round(goals_scored / len(team_matches), 3), round(goals_conceded / len(team_matches), 3)

def get_weighted_form(team, date, df, n=10):
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].sort_values('date').tail(n)
    
    if len(team_matches) < 5:
        return 0.5
    
    weighted_wins = 0
    total_weight = 0
    
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            opponent_elo = row['away_elo']
            won = row['outcome'] == 'Home win'
        else:
            opponent_elo = row['home_elo']
            won = row['outcome'] == 'Away win'
        
        weight = opponent_elo / 1500
        total_weight += weight
        if won:
            weighted_wins += weight
    
    return round(weighted_wins / total_weight, 3)

def get_days_since_last_match(team, date, df):
    prev_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].sort_values('date')
    
    if len(prev_matches) == 0:
        return 30
    
    last_match_date = prev_matches.iloc[-1]['date']
    days = (date - last_match_date).days
    return min(days, 90)

def predict_match(home_team, away_team, neutral, model, final_elo, df, feature_cols):
    current_date = pd.Timestamp('2026-01-01')
    
    home_form = get_recent_form(home_team, current_date, df)
    away_form = get_recent_form(away_team, current_date, df)
    home_weighted = get_weighted_form(home_team, current_date, df)
    away_weighted = get_weighted_form(away_team, current_date, df)
    home_h2h = get_h2h_record(home_team, away_team, current_date, df)
    away_h2h = 1 - home_h2h
    h2h_draw = get_h2h_draw_rate(home_team, away_team, current_date, df)
    home_scored, home_conceded = get_goal_averages(home_team, current_date, df)
    away_scored, away_conceded = get_goal_averages(away_team, current_date, df)
    home_goal_diff = home_scored - home_conceded
    away_goal_diff = away_scored - away_conceded
    home_elo = final_elo.get(home_team, 1500)
    away_elo = final_elo.get(away_team, 1500)
    elo_diff = home_elo - away_elo
    elo_closeness = 1 / (1 + abs(elo_diff))
    form_closeness = 1 / (1 + abs(home_form - away_form))
    home_rest = get_days_since_last_match(home_team, current_date, df)
    away_rest = get_days_since_last_match(away_team, current_date, df)
    combined_attack = home_scored + away_scored
    combined_defence = home_conceded + away_conceded
    expected_goals = (combined_attack + combined_defence) / 2
    home_advantage = 0 if neutral else 1

    features = [[
        home_elo, away_elo, elo_diff,
        home_form, away_form,
        home_weighted, away_weighted,
        home_h2h, away_h2h, h2h_draw,
        home_scored, home_conceded,
        away_scored, away_conceded,
        home_goal_diff, away_goal_diff,
        home_rest, away_rest,
        home_advantage,
        elo_closeness, form_closeness,
        combined_attack, combined_defence,
        expected_goals
    ]]

    probs = model.predict_proba(features)[0]
    classes = model.classes_

    result = {}
    for cls, prob in zip(classes, probs):
        result[cls] = round(float(prob) * 100, 1)

    result['home_elo'] = round(home_elo)
    result['away_elo'] = round(away_elo)
    result['home_form'] = home_form
    result['away_form'] = away_form

    return result