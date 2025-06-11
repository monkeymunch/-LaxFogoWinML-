import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# web scraper functions for input data
def get_faceoff_data(url="https://www.ncaa.com/stats/lacrosse-men/d1/current/individual/410"):
    """scrape faceoff statistics from ncaa website for individual players."""
    data, pg = [], 1
    while pg <= 2:
        soup = BeautifulSoup(requests.get(f"{url}/p{pg}" if pg > 1 else url).text, "html.parser")
        table = soup.find("table")
        if not table: break
        for row in table.find("tbody").find_all("tr"):
            cols = [td.text.strip() for td in row.find_all("td")]
            if len(cols) < 11 or "FO" not in cols[4]: continue
            try:
                data.append({
                    "Player": cols[1], "Team": cols[2],
                    "FO Wins": int(cols[6]), "FO Losses": int(cols[7]),
                    "FO Attempts": int(cols[8]),
                    "FO Percentage": float(cols[10]) if cols[10].startswith("0.") else float("0" + cols[10])
                })
            except: continue
        pg += 1
    return pd.DataFrame(data)

def get_gb_data(url="https://www.ncaa.com/stats/lacrosse-men/d1/current/individual/227"):
    """scrape ground ball statistics from ncaa website for midfielders and lsm players."""
    data, pg = [], 1
    while pg <= 4:
        soup = BeautifulSoup(requests.get(f"{url}/p{pg}" if pg > 1 else url).text, "lxml")
        table = soup.find("table")
        if not table: break
        for row in table.find("tbody").find_all("tr"):
            cols = [td.text.strip() for td in row.find_all("td")]
            if len(cols) < 8 or cols[4] not in ["M", "LSM"]: continue
            try:
                data.append({"Player": cols[1], "Team": cols[2], "GB Per Game": float(cols[7])})
            except: continue
        pg += 1
    return pd.DataFrame(data)

def get_cto_data(url="https://www.ncaa.com/stats/lacrosse-men/d1/current/individual/560"):
    """scrape caused turnovers statistics from ncaa website for midfielders and lsm players."""
    data, pg = [], 1
    while pg <= 5:
        soup = BeautifulSoup(requests.get(f"{url}/p{pg}" if pg > 1 else url).text, "lxml")
        table = soup.find("table")
        if not table: break
        for row in table.find("tbody").find_all("tr"):
            cols = [td.text.strip() for td in row.find_all("td")]
            if len(cols) < 8 or cols[4] not in ["M", "LSM"]: continue
            try:
                data.append({"Player": cols[1], "Team": cols[2], "CT Per Game": float(cols[7])})
            except: continue
        pg += 1
    return pd.DataFrame(data)

def get_clear_pct(url="https://www.ncaa.com/stats/lacrosse-men/d1/current/team/838"):
    """scrape team clearing percentage statistics from ncaa website."""
    data, pg = [], 1
    while pg <= 2:
        soup = BeautifulSoup(requests.get(f"{url}/p{pg}" if pg > 1 else url).text, "lxml")
        table = soup.find("table")
        if not table: break
        for row in table.find("tbody").find_all("tr"):
            cols = [td.text.strip() for td in row.find_all("td")]
            if len(cols) < 6: continue
            try:
                data.append({"Team": cols[1], "Clearing Pct": float(cols[5]) if cols[5].startswith("0.") or cols[5].startswith("1.") else float("0" + cols[5])})
            except: continue
        pg += 1
    return pd.DataFrame(data)

def get_laxnumbers_ratings():
    """scrape team power ratings from laxnumbers website using selenium webdriver."""
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.laxnumbers.com/ratings.php?y=2025&v=401")

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "rankings-table-body"))
        )
    except:
        html = driver.page_source
        driver.quit()
        return pd.DataFrame(), html

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    data = []
    for row in soup.select(".rankings-table-body tr"):
        cols = row.find_all("td")
        if len(cols) < 6: continue
        try:
            data.append({
                "Team": cols[1].text.strip(),
                "Rank": int(cols[0].text.strip()),
                "Power": float(cols[3].text.strip()),
                "SoS": float(cols[5].text.strip())
            })
        except: continue
    return pd.DataFrame(data), None

def compute_wing_boost(gb_df, ct_df):
    """calculate wing boost scores based on ground balls and caused turnovers performance."""
    gb_team = gb_df.groupby("Team")["GB Per Game"].sum().reset_index(name="GB Score")
    ct_team = ct_df.groupby("Team")["CT Per Game"].sum().reset_index(name="CT Score")
    combo = pd.merge(gb_team, ct_team, on="Team", how="outer").fillna(0)
    combo["WScore"] = 0.7 * combo["GB Score"] + 0.3 * combo["CT Score"]
    combo["Rank"] = combo["WScore"].rank(ascending=False, method='min')
    combo["Boost"] = combo["Rank"].apply(lambda r: 0.04 if r <= 10 else 0.025)
    return dict(zip(combo["Team"], combo["Boost"]))

# === Synthetic Training Data Generation === #
def generate_synthetic_labels(matchups_df):
    """generate more realistic labels based on team strength differences with noise."""
    labels = []
    
    for _, row in matchups_df.iterrows():
        power_diff = row['Power'] - row['Opp Power']
        rank_diff = row['Opp Rank'] - row['Rank'] 
        fo_diff = row['FO%'] - row['Opp FO%']
        clear_diff = row['Clearing%'] - row['Opp Clearing%']
        
        strength_score = (
            0.4 * power_diff +
            0.3 * rank_diff / 50 +  
            0.2 * fo_diff * 100 +   # Scale FO% difference
            0.1 * clear_diff * 100  
        )
        
        # convert probability using sigmoid curve
        win_prob = 1 / (1 + np.exp(-strength_score / 2))
        
        # add random aspect
        noise = np.random.normal(0, 0.1)
        win_prob = np.clip(win_prob + noise, 0.05, 0.95)
        
        # generate binary outcome
        label = 1 if np.random.random() < win_prob else 0
        labels.append(label)
    
    return labels

def get_stat(d, key, default):
    """get statistic from dictionary with fallback to default value."""
    return d.get(key, default)

def get_rating_stat(team, field, default):
    """get rating statistic for team with fuzzy matching and fallback to default."""
    if team in ratings_dict:
        return ratings_dict[team].get(field, default)
    for name in ratings_dict:
        if team.lower() in name.lower():
            return ratings_dict[name].get(field, default)
    return default

def create_prediction_row(team_a, team_b):
    """create feature row for predicting team_a vs team_b matchup."""
    row = {
        "FO%": get_stat(fo_pct, team_a, fo_mean),
        "Opp FO%": get_stat(fo_pct, team_b, fo_mean),
        "Clearing%": get_stat(clear_pct, team_a, clear_mean),
        "Opp Clearing%": get_stat(clear_pct, team_b, clear_mean),
        "Wing Boost": get_stat(wing_boost, team_a, 0),
        "Opp Wing Boost": get_stat(wing_boost, team_b, 0),
        "Power": get_rating_stat(team_a, "Power", power_mean),
        "Opp Power": get_rating_stat(team_b, "Power", power_mean),
        "SoS": get_rating_stat(team_a, "SoS", sos_mean),
        "Opp SoS": get_rating_stat(team_b, "SoS", sos_mean),
        "Rank": get_rating_stat(team_a, "Rank", rank_mean),
        "Opp Rank": get_rating_stat(team_b, "Rank", rank_mean),
        }
    row_df = pd.DataFrame([row], columns=X.columns)  # make sure we have valid feature names
    row_scaled = scaler.transform(row_df)
    return row_scaled, row

# prepare input data
fo_df = get_faceoff_data()
gb_df = get_gb_data()
cto_df = get_cto_data()
clear_df = get_clear_pct()
ratings_df, raw_html = get_laxnumbers_ratings()


wing_boost = compute_wing_boost(gb_df, cto_df)

fo_pct = fo_df.groupby("Team")["FO Percentage"].mean().to_dict()
clear_pct = clear_df.set_index("Team")["Clearing Pct"].to_dict()
ratings_dict = ratings_df.set_index("Team")[["Power", "SoS", "Rank"]].to_dict(orient="index")

fo_mean = np.mean(list(fo_pct.values()))
clear_mean = np.mean(list(clear_pct.values()))
power_mean = np.mean([v["Power"] for v in ratings_dict.values()])
sos_mean = np.mean([v["SoS"] for v in ratings_dict.values()])
rank_mean = np.mean([v["Rank"] for v in ratings_dict.values()])
teams = sorted(set(fo_pct) | set(clear_pct) | set(wing_boost) | set(ratings_dict))

# Create all the features for input data
features = []
for a in teams:
    for b in teams:
        if a == b: continue
        features.append({
            "FO%": get_stat(fo_pct, a, fo_mean),
            "Opp FO%": get_stat(fo_pct, b, fo_mean),
            "Clearing%": get_stat(clear_pct, a, clear_mean),
            "Opp Clearing%": get_stat(clear_pct, b, clear_mean),
            "Wing Boost": get_stat(wing_boost, a, 0),
            "Opp Wing Boost": get_stat(wing_boost, b, 0),
            "Power": get_rating_stat(a, "Power", power_mean),
            "Opp Power": get_rating_stat(b, "Power", power_mean),
            "SoS": get_rating_stat(a, "SoS", sos_mean),
            "Opp SoS": get_rating_stat(b, "SoS", sos_mean),
            "Rank": get_rating_stat(a, "Rank", rank_mean),
            "Opp Rank": get_rating_stat(b, "Rank", rank_mean),
        })

data_df = pd.DataFrame(features)

np.random.seed(42)  
synthetic_labels = generate_synthetic_labels(data_df)
data_df["Label"] = synthetic_labels

X = data_df.drop(columns=["Label"])
y = data_df["Label"]
#  scale the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# logistic regression binary classification faceoff by faceoff, game by game
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print(f"Model accuracy: {model.score(X_test, y_test):.3f}")

# user input prediction
print("\nFaceoff Predictor Ready. Type two team names:")
t1 = input("Team 1: ").strip()
t2 = input("Team 2: ").strip()

if t1 in teams and t2 in teams:
    row1_scaled, row1_raw = create_prediction_row(t1, t2)
    row2_scaled, row2_raw = create_prediction_row(t2, t1)

    prob1 = model.predict_proba(row1_scaled)[0][1]
    prob2 = model.predict_proba(row2_scaled)[0][1]
    
    print(f"\n{t1} win chance vs {t2}: {prob1:.1%}")
    print(f"{t2} win chance vs {t1}: {prob2:.1%}")
    
    # Show key stat differences
    fo_diff = row1_raw['FO%'] - row1_raw['Opp FO%']
    power_diff = row1_raw['Power'] - row1_raw['Opp Power']
    rank_diff = row1_raw['Opp Rank'] - row1_raw['Rank']
    
    print(f"\nKey Differences ({t1} vs {t2}):")
    print(f"  FO% Advantage: {fo_diff:+.1%}")
    print(f"  Power Rating Advantage: {power_diff:+.1f}")
    print(f"  Rank Advantage: {rank_diff:+.0f} positions")
    
else:
    print(f"\nAvailable teams: {', '.join(teams[:10])}...")
    print("Check spelling and try again.")

# generate heatmap
print("\nGenerating win probability heatmap...")
heatmap = pd.DataFrame(index=teams, columns=teams, dtype=float)

for i, team_a in enumerate(teams):
    for j, team_b in enumerate(teams):
        if team_a == team_b:
            heatmap.loc[team_a, team_b] = np.nan
            continue
        try:
            row_scaled, _ = create_prediction_row(team_a, team_b)
            win_prob = model.predict_proba(row_scaled)[0][1]
            heatmap.loc[team_a, team_b] = round(win_prob, 3)
        except Exception as e:
            print(f"Error for {team_a} vs {team_b}: {e}")
            heatmap.loc[team_a, team_b] = np.nan

plt.figure(figsize=(20, 16))
sns.heatmap(heatmap.astype(float), cmap="RdYlBu_r", center=0.5, 
            cbar_kws={'label': 'Win Probability'}, 
            annot=False, fmt='.2f')
plt.title("Predicted Win Probability Heatmap\n(Row Team vs Column Team)", fontsize=16)
plt.xlabel("Opponent", fontsize=12)
plt.ylabel("Team", fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()