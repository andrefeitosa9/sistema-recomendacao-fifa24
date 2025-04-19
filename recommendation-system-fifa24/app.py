import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

# Load data (cached for efficiency)
@st.cache_data
def load_data():
    df = pd.read_parquet("data/silver.parquet")
    return df

df = load_data()

# Define rating attributes for each position
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

possible_positions = ['GK', 'CB', 'LB', 'RB', 'CDM', 'LWB', 'RWB', 'LM', 'RM', 'CAM', 'CM', 'LW', 'RW', 'ST', 'CF']
position_ratings_map = {
    'GK': GK_ratings,
    'CB': CB_ratings,
    'LB': LB_ratings,
    'RB': RB_ratings,
    'CDM': CDM_ratings,
    'LWB': LWB_ratings,
    'RWB': RWB_ratings,
    'LM': LM_ratings,
    'RM': RM_ratings,
    'CAM': CAM_ratings,
    'CM': CM_ratings,
    'LW': LW_ratings,
    'RW': RW_ratings,
    'ST': ST_ratings,
    'CF': CF_ratings,
}

def prepare_player_data_for_recommendation(df, position_ref, columns_player, filters=None):
    if position_ref in position_ratings_map:
        ratings_position = position_ratings_map[position_ref]
        df_columns = columns_player + ratings_position
        df_filtered = df[df['player_positions'].str.contains(position_ref, case=False, na=False)].copy()
        df_filtered = df_filtered[df_columns].reset_index(drop=True)
        features = ratings_position

        if filters:
            for col, value in filters.items():
                if value is not None and col in df_filtered.columns:
                    if isinstance(value, tuple):
                        df_filtered = df_filtered[df_filtered[col].between(value[0], value[1])]
                    else:
                        df_filtered = df_filtered[df_filtered[col] == value]
        return df_filtered, features
    else:
        st.error(f"Position '{position_ref}' is not valid.")
        return pd.DataFrame(), []

st.title("FIFA 24 Player Recommendation System")

# --- User Input for Reference Player ---
st.subheader("Select a Reference Player")
player_names = df['short_name'].unique()
selected_player_name = st.selectbox("Search or select a player:", sorted(player_names))
reference_player_id = df[df['short_name'] == selected_player_name]['player_id'].iloc[0]
st.info(f"Reference Player ID: {reference_player_id}")

# --- User Selection for Target Position ---
selected_position = st.selectbox("Select Target Position:", possible_positions)

# --- User Input for Filters ---
st.sidebar.header("Optional Filters")
overall_filter = st.sidebar.slider("Overall Range:", 0, 100, (0, 100))
age_filter = st.sidebar.slider("Age Range:", 16, 40, (16, 40))
nationality_filter = st.sidebar.text_input("Filter by Nationality (optional):")

filters = {
    'overall': overall_filter,
    'age': age_filter,
    'nationality_name': nationality_filter if nationality_filter else None,
}

if st.button("Get Recommendations"):
    df_model, features = prepare_player_data_for_recommendation(df, selected_position, columns_player, filters)

    if not df_model.empty and features:
        if reference_player_id in df_model['player_id'].values:
            reference_player_data = df_model[df_model['player_id'] == reference_player_id][features].values
            if reference_player_data.shape[0] > 0:
                reference_player_data_single = reference_player_data[0].reshape(1, -1)

                # Handle potential infinite values
                df_model_numerical = df_model[features].replace([np.inf, -np.inf], np.nan).dropna()

                if not df_model_numerical.empty:
                    scaler = StandardScaler()
                    normalized_data = scaler.fit_transform(df_model_numerical)

                    knn = NearestNeighbors(n_neighbors=11, metric='cosine') # Get top 10 recommendations + self
                    knn.fit(normalized_data)

                    # Find the normalized data of the reference player
                    reference_player_df = df_model[df_model['player_id'] == reference_player_id]
                    if not reference_player_df.empty:
                        reference_player_features = reference_player_df[features].replace([np.inf, -np.inf], np.nan).dropna()
                        if not reference_player_features.empty:
                            reference_player_normalized = scaler.transform(reference_player_features)
                            distances, indices = knn.kneighbors(reference_player_normalized)

                            recommended_players_df = df_model_numerical.iloc[indices[0][1:]] # Exclude the reference player
                            recommended_player_ids = df_model.iloc[recommended_players_df.index]['player_id'].tolist()
                            recommendations = df[df['player_id'].isin(recommended_player_ids)][['short_name', 'long_name', 'player_positions', 'overall', 'club_name']]

                            st.subheader("Recommended Players:")
                            st.dataframe(recommendations)

                            # Visualization (Radar Chart for top 4 recommendations + reference)
                            if not recommendations.empty:
                                top_recommendations_ids = recommendations['player_id'].tolist()[:4]
                                players_for_chart_ids = [reference_player_id] + top_recommendations_ids
                                players_for_chart = df_model[df_model['player_id'].isin(players_for_chart_ids)]

                                if not players_for_chart.empty:
                                    num_features = len(features)
                                    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
                                    angles += angles[:1]

                                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

                                    for _, player in players_for_chart.iterrows():
                                        values = player[features].fillna(0).values.tolist() # Handle NaN for plotting
                                        values += values[:1]
                                        ax.plot(angles, values, linewidth=2, label=player['short_name'])
                                        ax.fill(angles, values, alpha=0.25)

                                    ax.set_xticks(angles[:-1])
                                    ax.set_xticklabels(features, fontsize=8, rotation=45)
                                    ax.set_yticklabels([])
                                    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=8)
                                    plt.title(f'Feature Comparison with Reference Player', fontsize=12, y=1.1)
                                    st.pyplot(fig)
                                else:
                                    st.warning("Could not prepare data for the radar chart.")

                        else:
                            st.warning("Reference player has missing feature data.")
                    else:
                        st.warning("No numerical data available for the selected features after handling missing values.")
            else:
                st.warning(f"No feature data found for reference player ID {reference_player_id}.")
        else:
            st.info("Please select a position and click 'Get Recommendations'.")