import json
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load champion JSON data
with open('champion_info.json', 'r') as file:
    champion_data = json.load(file)

# Create ID-to-name and name-to-ID mappings
id_to_name = {int(champ_id): champ_info['name'] for champ_id, champ_info in champion_data['data'].items()}
name_to_id = {champ_info['name']: int(champ_id) for champ_id, champ_info in champion_data['data'].items()}
all_champions = list(name_to_id.keys())  # List of all champion names

# Load your dataset
data = pd.read_csv('games.csv')

# Prepare features and target
feature_columns = [
    't1_champ1id', 't1_champ2id', 't1_champ3id', 't1_champ4id', 't1_champ5id',
    't2_champ1id', 't2_champ2id', 't2_champ3id', 't2_champ4id', 't2_champ5id'
]
X = data[feature_columns]
y = (data['winner'] == 1).astype(int)  # Convert 'winner' to binary (1 if T1 wins, 0 otherwise)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.values, y_train)

# Define the recommendation function
def recommend_top_5_champions_with_names(team1_champs_locked, team2_champs):
    # Convert champion names to IDs
    team1_champs_locked_ids = [name_to_id[name] for name in team1_champs_locked]
    team2_champs_ids = [name_to_id[name] for name in team2_champs]

    # Determine the set of champions already picked
    picked_champions_ids = set(team1_champs_locked_ids + team2_champs_ids)

    # Find all unpicked champions
    all_champions_ids = set(name_to_id[name] for name in all_champions)
    available_champions_ids = all_champions_ids - picked_champions_ids

    # Store probabilities for each available champion
    champion_probabilities = []

    for champ_id in available_champions_ids:
        # Construct the row representing the full game state if you pick 'champ_id'
        row = team1_champs_locked_ids + [champ_id] + team2_champs_ids

        # Convert to a shape (1, 10) for prediction
        row = np.array(row).reshape(1, -1)

        # Predict probability that T1 wins
        prob_win = model.predict_proba(row)[0, 1]

        # Store the champion ID and its predicted probability
        champion_probabilities.append((champ_id, prob_win))

    # Sort the champions by predicted win probability in descending order
    champion_probabilities.sort(key=lambda x: x[1], reverse=True)

    # Get the top 5 champions and convert IDs back to names
    top_5_champions = [(id_to_name[champ_id], prob) for champ_id, prob in champion_probabilities[:5]]

    return top_5_champions

# Streamlit UI
st.title("League of Legends Champion Recommendation")
st.markdown("<h4 style='color: red;'>WIP: Using outdated League patch</h4>", unsafe_allow_html=True)

# Default teams
default_team1 = ['Ahri', 'Jinx', 'Thresh', 'Lee Sin']
default_team2 = ['Zed', 'Lux', 'Draven', 'Vi', 'Morgana']

# Team 1 champion selection
st.header("Team 1 Champions")
team1_champs = []
for i in range(4):
    champ = st.selectbox(f"T1 Champ {i+1}", all_champions, index=all_champions.index(default_team1[i]))
    team1_champs.append(champ)

# Team 2 champion selection
st.header("Team 2 Champions")
team2_champs = []
for i in range(5):
    champ = st.selectbox(f"T2 Champ {i+1}", all_champions, index=all_champions.index(default_team2[i]))
    team2_champs.append(champ)

# Validate unique picks
if len(set(team1_champs + team2_champs)) != len(team1_champs + team2_champs):
    st.error("Error: Duplicate champions detected. Please ensure each champion is picked only once.")
else:
    # Recommendation button
    if st.button("Recommend Champions"):
        recommendations = recommend_top_5_champions_with_names(team1_champs, team2_champs)
        st.subheader("Top 5 Recommended Champions:")
        for champ_name, winrate in recommendations:
            st.write(f"{champ_name}: {winrate:.2%}")
