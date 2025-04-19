import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os


# Loading silver data
df = pd.read_parquet(r"data/silver.parquet")

# Reference Player and Position
reference_player_id = 238794
reference_position = "LW"
overall_threshold = None
potential_threshold = None
value_eur_threshold = None
wage_eur_threshold = None
age_threshold = None
contract_until_year_threshold = None
nationality_threshold = None
release_clause_threshold = None
player_tags_threshold = None
player_traits_threshold = None


GK_ratings = ['skill_long_passing', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
                    'power_shot_power', 'power_jumping', 'power_strength', 'mentality_vision', 'mentality_composure',
                    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes',
                    'goalkeeping_speed']


CB_ratings = ['pace', 'passing', 'dribbling', 'defending', 'physic', 'attacking_heading_accuracy', 'attacking_short_passing', 'skill_long_passing',
                    'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_reactions', 'power_jumping', 'power_stamina',
                    'power_strength', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_composure',
                    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle', 'attacking_work_rate', 'defending_work_rate']


LB_ratings = ['pace', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_short_passing', 'skill_curve',
                    'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                    'power_stamina', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
                    'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
                    'attacking_work_rate', 'defending_work_rate']

RB_ratings = list(LB_ratings)


CDM_ratings = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_heading_accuracy', 'attacking_short_passing',
                    'skill_dribbling', 'skill_long_passing', 'movement_acceleration', 'movement_agility', 'movement_reactions', 'movement_balance',
                    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
                    'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_composure', 'defending_marking_awareness',
                    'defending_standing_tackle', 'defending_sliding_tackle', 'attacking_work_rate', 'defending_work_rate']


LWB_ratings = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_short_passing', 'skill_dribbling',
                    'skill_curve', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                    'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
                    'mentality_positioning', 'mentality_vision', 'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
                    'attacking_work_rate', 'defending_work_rate']

RWB_ratings = list(LWB_ratings)
LM_ratings = list(LWB_ratings)
RM_ratings = list(LWB_ratings)


CAM_ratings = ['pace', 'shooting', 'passing', 'dribbling', 'physic', 'attacking_finishing', 'attacking_short_passing', 'skill_dribbling', 'skill_curve',
                    'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_agility', 'movement_reactions',
                    'movement_balance', 'power_shot_power', 'power_stamina', 'power_strength', 'power_long_shots',
                    'mentality_positioning', 'mentality_vision', 'mentality_composure', 'attacking_work_rate', 'defending_work_rate']

CM_ratings = list(CAM_ratings)


LW_ratings = ['pace', 'shooting', 'passing', 'dribbling', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_short_passing', 'attacking_volleys',
                    'skill_dribbling', 'skill_curve', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                    'movement_balance', 'power_shot_power', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_positioning', 'mentality_vision',
                    'mentality_composure', 'attacking_work_rate', 'defending_work_rate']

RW_ratings = list(LW_ratings)


ST_ratings = ['pace', 'shooting', 'passing', 'dribbling', 'physic', 'attacking_finishing', 'attacking_heading_accuracy',
                    'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_ball_control', 'movement_acceleration',
                    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
                    'power_stamina', 'power_strength', 'power_long_shots', 'mentality_positioning', 'mentality_vision',
                    'mentality_composure', 'attacking_work_rate', 'defending_work_rate']


CF_ratings = list(ST_ratings)

columns_player = ['player_id', 'short_name', 'long_name', 'player_positions', 'overall', 'potential', 'value_eur', 'wage_eur',
                    'age', 'club_name', 'league_name', 'club_contract_valid_until_year', 'nationality_name',
                    'release_clause_eur', 'player_tags', 'player_traits']


def prepare_player_data_and_filter(
    df,
    position_ref,
    columns_player,
    overall_threshold=None,
    potential_threshold=None,
    value_eur_threshold=None,
    wage_eur_threshold=None,
    age_threshold=None,
    contract_until_year_threshold=None,
    nationality_threshold=None,
    release_clause_threshold=None,
    player_tags_threshold=None,
    player_traits_threshold=None
):
    """
    Prepares player data by filtering based on position and other criteria,
    selecting relevant columns based on the position.

    Args:
        df (pd.DataFrame): DataFrame containing player data.
        position_ref (str): Reference position to filter players.
        columns_player (list): List of base player columns.
        overall_threshold (tuple, optional): Range (min, max) to filter by overall. Defaults to None.
        potential_threshold (tuple, optional): Range to filter by potential. Defaults to None.
        value_eur_threshold (tuple, optional): Range to filter by value in euros. Defaults to None.
        wage_eur_threshold (tuple, optional): Range to filter by wage in euros. Defaults to None.
        age_threshold (tuple, optional): Range to filter by age. Defaults to None.
        contract_until_year_threshold (int, optional): Contract validity year to filter. Defaults to None.
        nationality_threshold (str, optional): Nationality name to filter. Defaults to None.
        release_clause_threshold (tuple, optional): Range to filter by release clause. Defaults to None.
        player_tags_threshold (list, optional): List of tags to filter (player must contain all). Defaults to None.
        player_traits_threshold (list, optional): List of traits to filter (player must contain all). Defaults to None.

    Returns:
        tuple: A filtered DataFrame and a list of feature columns specific to the position.
               Returns an empty DataFrame and an empty list if the position is not found.
    """
    if position_ref + '_ratings' in globals():
        ratings_position = globals()[position_ref + '_ratings']
        df_columns = columns_player + ratings_position
        df_filtered = df[df['player_positions'].str.contains(position_ref, case=False, na=False)].copy()
        df_filtered = df_filtered[df_columns].reset_index(drop=True)
        features = ratings_position

        # Apply numerical range filters
        range_filters = {
            'overall': overall_threshold,
            'potential': potential_threshold,
            'value_eur': value_eur_threshold,
            'wage_eur': wage_eur_threshold,
            'age': age_threshold,
            'club_contract_valid_until_year': contract_until_year_threshold,
            'release_clause_eur': release_clause_threshold
        }

        for column, threshold in range_filters.items():
            if threshold is not None and column in df_filtered.columns:
                minimum = threshold[0] if isinstance(threshold, tuple) and threshold[0] is not None else -np.inf
                maximum = threshold[1] if isinstance(threshold, tuple) and threshold[1] is not None else np.inf
                df_filtered = df_filtered[df_filtered[column].between(minimum, maximum)]
            elif isinstance(threshold, (int, float)) and column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column] == threshold]

        # Filter by nationality
        if nationality_threshold is not None and 'nationality_name' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['nationality_name'] == nationality_threshold]

        # Filter by tags
        if player_tags_threshold is not None and 'player_tags' in df_filtered.columns:
            for tag in player_tags_threshold:
                df_filtered = df_filtered[df_filtered['player_tags'].str.contains(tag, case=False, na=False)]

        # Filter by traits
        if player_traits_threshold is not None and 'player_traits' in df_filtered.columns:
            for trait in player_traits_threshold:
                df_filtered = df_filtered[df_filtered['player_traits'].str.contains(trait, case=False, na=False)]

        return df_filtered, features
    else:
        print(f"No rating attributes defined for position '{position_ref}'.")
        return pd.DataFrame(), []


df_model, features = prepare_player_data_and_filter(
    df=df,
    position_ref=reference_position,
    columns_player=columns_player
)

if df_model.empty or not features:
    print("No data available for the specified position. Exiting.")
else:
    # Model training
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_model[features])

    knn = NearestNeighbors(n_neighbors=len(df_model), metric='cosine', algorithm='brute')  # n_neighbors = total of players
    knn.fit(normalized_data)

    # Get the index of the reference player
    try:
        idx_ref = df_model[df_model['player_id'] == reference_player_id].index[0]

        # Get the indices and distances for all players
        distances, indices = knn.kneighbors([normalized_data[idx_ref]])

        # Create a DataFrame with the results
        df_results = pd.DataFrame({
            'Player_ID': df_model.iloc[indices[0]]['player_id'].values,
            'Name': df_model.iloc[indices[0]]['short_name'].values,
            'Full Name': df_model.iloc[indices[0]]['long_name'].values,
            'Position': df_model.iloc[indices[0]]['player_positions'].values,
            'Overall': df_model.iloc[indices[0]]['overall'].values,
            'Similarity': 1 - distances[0]  # Similarity = 1 - cosine distance
        })

        # Exclude the reference player and sort by similarity in descending order
        results = df_results[df_results['Player_ID'] != reference_player_id].sort_values(by='Similarity', ascending=False)

        # Select the top 100 most similar players
        top_100 = results.head(100)

        print(top_100)

        reference_player_name = df_model[df_model['player_id'] == reference_player_id]['short_name'].values[0]

        top_4 = results.head(4)

        # Add the reference player to the DataFrame for the chart
        players_for_chart = pd.concat([df_model[df_model['player_id'] == reference_player_id], df_model[df_model['player_id'].isin(top_4['Player_ID'])]])

        # Radar chart
        num_features = len(features)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle (adding the first value back to the end)

        # Create the chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each player on the radar chart
        for _, player in players_for_chart.iterrows():
            values = player[features].values.tolist()
            values += values[:1]  # Close the circle
            ax.plot(angles, values, linewidth=2, label=player['short_name'])
            ax.fill(angles, values, alpha=0.25)

        # Chart adjustments
        ax.set_xticks(angles[:-1])  # Avoid the last repetition of the angle
        ax.set_xticklabels(features, fontsize=10, rotation=45)  # Adjust the font
        ax.set_yticklabels([])  # Remove the y-axis labels

        # Legend and title
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=12)
        plt.title(f'Feature Comparison: {reference_player_name} and Recommended', fontsize=16)

        # Display the chart
        plt.tight_layout()
        plt.show()

    except IndexError:
        print(f"Reference player with ID {reference_player_id} not found in the filtered data for position {reference_position}.")