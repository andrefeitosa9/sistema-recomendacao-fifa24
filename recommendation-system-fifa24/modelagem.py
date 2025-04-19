import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os


# Loading silver data
df = pd.read_parquet(r"data\silver.parquet")

# Jogador e Posição referência
id_player_ref = 238794
position_ref = "LW"
overall = None
potential_ref = None
value_eur = None
wage_eur = None
age = None
club_contract_valid_until_year = None
nationality_name = None
release_clause_eur = None
player_tags = None
player_traits = None


GK = ['skill_long_passing', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
                   'power_shot_power', 'power_jumping', 'power_strength', 'mentality_vision', 'mentality_composure',
                   'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes',
                   'goalkeeping_speed']




CB = ['pace', 'passing', 'dribbling', 'defending', 'physic', 'attacking_heading_accuracy', 'attacking_short_passing', 'skill_long_passing',
                   'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_reactions', 'power_jumping', 'power_stamina',
                   'power_strength', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_composure',
                   'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle', 'attacking_work_rate', 'defending_work_rate']




LB = ['pace', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_short_passing', 'skill_curve', 
                   'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                   'power_stamina', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
                   'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
                   'attacking_work_rate', 'defending_work_rate']

RB = list(LB)




CDM = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_heading_accuracy', 'attacking_short_passing',
                   'skill_dribbling', 'skill_long_passing', 'movement_acceleration', 'movement_agility', 'movement_reactions', 'movement_balance',
                   'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
                   'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_composure', 'defending_marking_awareness',
                   'defending_standing_tackle', 'defending_sliding_tackle', 'attacking_work_rate', 'defending_work_rate']




LWB = ['pace', 'shooting', 'passing',  'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_short_passing', 'skill_dribbling',
            'skill_curve', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
            'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
            'mentality_positioning', 'mentality_vision', 'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
            'attacking_work_rate', 'defending_work_rate']

RWB = list(LWB)
LM = list(LWB)
RM = list(LWB)



CAM = ['pace', 'shooting', 'passing', 'dribbling', 'physic', 'attacking_finishing', 'attacking_short_passing', 'skill_dribbling', 'skill_curve',
                     'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_agility', 'movement_reactions',
                     'movement_balance', 'power_shot_power', 'power_stamina', 'power_strength', 'power_long_shots', 
                     'mentality_positioning', 'mentality_vision', 'mentality_composure', 'attacking_work_rate', 'defending_work_rate']

CM = list(CAM)


LW = ['pace', 'shooting', 'passing', 'dribbling', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_short_passing', 'attacking_volleys',
                   'skill_dribbling', 'skill_curve', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                   'movement_balance', 'power_shot_power', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_positioning', 'mentality_vision', 
                   'mentality_composure', 'attacking_work_rate', 'defending_work_rate']

RW = list(LW)


ST = ['pace', 'shooting', 'passing', 'dribbling', 'physic', 'attacking_finishing', 'attacking_heading_accuracy',
                'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_ball_control', 'movement_acceleration',
                'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
                'power_stamina', 'power_strength', 'power_long_shots', 'mentality_positioning', 'mentality_vision', 
                'mentality_composure', 'attacking_work_rate', 'defending_work_rate']


CF = list(ST)

columns_player = ['player_id', 'short_name', 'long_name', 'player_positions', 'overall', 'potential', 'value_eur', 'wage_eur',
                  'age', 'club_name', 'league_name', 'club_contract_valid_until_year', 'nationality_name',
                    'release_clause_eur',  'player_tags', 'player_traits']



def preparing_players_data_and_filtering_df(
    df,
    position_ref,
    columns_player,
    overall_ref=None,
    potencial_ref=None,
    value_eur_ref=None,
    wage_eur_ref=None,
    age_ref=None,
    club_contract_valid_until_year_ref=None,
    nationality_name_ref=None,
    release_clause_eur_ref=None,
    player_tags_ref=None,
    player_traits_ref=None
):
    if position_ref in globals():
        ratings_position = globals()[position_ref]
        df_columns = columns_player + ratings_position
        df_filtrado = df[df['player_positions'].str.contains(position_ref, case=False, na=False)].copy()
        df_filtrado = df_filtrado[df_columns].reset_index(drop=True)
        features = ratings_position

        # Aplica filtros numéricos com ranges
        filtros_range = {
            'overall': overall_ref,
            'potential': potencial_ref,
            'value_eur': value_eur_ref,
            'wage_eur': wage_eur_ref,
            'age': age_ref,
            'club_contract_valid_until_year': club_contract_valid_until_year_ref,
            'release_clause_eur': release_clause_eur_ref
        }

        for coluna, intervalo in filtros_range.items():
            if intervalo is not None and coluna in df_filtrado.columns:
                minimo = intervalo[0] if intervalo[0] is not None else -np.inf
                maximo = intervalo[1] if intervalo[1] is not None else np.inf
                df_filtrado = df_filtrado[df_filtrado[coluna].between(minimo, maximo)]

        # Filtro por nacionalidade
        if nationality_name_ref is not None and 'nationality_name' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['nationality_name'] == nationality_name_ref]

        # Filtro por tags
        if player_tags_ref is not None and 'player_tags' in df_filtrado.columns:
            for tag in player_tags_ref:
                df_filtrado = df_filtrado[df_filtrado['player_tags'].str.contains(tag, case=False, na=False)]

        # Filtro por traits
        if player_traits_ref is not None and 'player_traits' in df_filtrado.columns:
            for trait in player_traits_ref:
                df_filtrado = df_filtrado[df_filtrado['player_traits'].str.contains(trait, case=False, na=False)]

        return df_filtrado, features
    else:
        print(f"A variável ou a posição '{position_ref}' não foi definida ou não é válida.")
        return pd.DataFrame(), []
    

df_model, features = preparing_players_data_and_filtering_df(
    df = df,
    position_ref = position_ref,
    columns_player = columns_player)


# Model training


scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_model[features])


knn = NearestNeighbors(n_neighbors=len(df_model), metric='cosine', algorithm='brute')  # n_neighbors = total of players
knn.fit(normalized_data)

# Obter o índice do jogador de referência
idx_ref = df_model[df_model['player_id'] == id_player_ref].index[0]

# Obter os índices e distâncias para todos os jogadores
distancias, indices = knn.kneighbors([normalized_data[idx_ref]])

# Criar um DataFrame com os resultados
df_results = pd.DataFrame({
    'Player_ID': df_model.iloc[indices[0]]['player_id'].values,
    'Name': df_model.iloc[indices[0]]['short_name'].values,
    'Full Name': df_model.iloc[indices[0]]['long_name'].values,
    'Position': df_model.iloc[indices[0]]['player_positions'].values,
    'Overall': df_model.iloc[indices[0]]['overall'].values,
    'Similarity': 1 - distancias[0]  # Similaridade = 1 - distância de cosseno
})

# Excluir o jogador de referência e ordenar pelos valores de similaridade em ordem decrescente
resultados = df_results[df_results['Player_ID'] != id_player_ref].sort_values(by='Similarity', ascending=False)


# Selecionar os 100 jogadores mais similares
top_100 = resultados.head(100)

print(top_100)

jogador_ref = df_model[df_model['player_id'] == id_player_ref]['short_name'].values[0]


top_4 = resultados.head(4)

# Adicionar o jogador de referência ao DataFrame para o gráfico
jogadores_para_grafico = pd.concat([df_model[df_model['player_id'] == id_player_ref], df_model[df_model['player_id'].isin(top_4['Player_ID'])]])

# Gráfico de radar
num_features = len(features)
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
angles += angles[:1]  # Fechar o círculo (adicionando o primeiro valor de volta ao final)

# Criar o gráfico
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Plotar cada jogador no gráfico de radar
for _, jogador in jogadores_para_grafico.iterrows():
    valores = jogador[features].values.tolist()
    valores += valores[:1]  # Fechar o círculo
    ax.plot(angles, valores, linewidth=2, label=jogador['short_name'])
    ax.fill(angles, valores, alpha=0.25)

# Ajustes do gráfico
ax.set_xticks(angles[:-1])  # Evitar a última repetição do ângulo
ax.set_xticklabels(features, fontsize=10, rotation=45)  # Ajustar a fonte
ax.set_yticklabels([])  # Remover os rótulos do eixo Y

# Legenda e título
ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=12)
plt.title(f'Comparação de características: {jogador_ref} e recomendados', fontsize=16)

# Exibir o gráfico
plt.tight_layout()
plt.show()