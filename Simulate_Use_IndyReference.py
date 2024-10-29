import random
import pandas as pd

# ロボットができることリスト
abilities = ["案内", "おしゃべり", "手伝い", "スケジュール確認"]

# 会話内容の選択スコアと実行受け入れ回数を保管するDataFrame
columns = ["場所", "ユーザ活動", "できること", "選択スコア", "受け入れ回数"]
df_scores = pd.DataFrame(columns=columns)

# できることの使用回数を管理する辞書
usage_counts = {ability: 0 for ability in abilities}

# 場所とユーザ活動のリスト
locations = ["bath_room", "bed_room", "living", "kitchen", "landly"]
user_activities = ["use_internet", "return_from_others", "rest_relax", "cleaning"]

# 初期化関数
def initialize_scores():
    global df_scores
    for location in locations:
        for activity in user_activities:
            for ability in abilities:
                df_scores = df_scores.append({
                    "場所": location,
                    "ユーザ活動": activity,
                    "できること": ability,
                    "選択スコア": 1,  # 初期スコアを1に設定
                    "受け入れ回数": 0
                }, ignore_index=True)

# 話しかけるかどうかの判断
def should_initiate_conversation(location, activity, threshold=0.5):
    timing_score = random.uniform(0, 1)  # タイミングスコアをランダムに生成
    print("timeing_score: ", timing_score)
    return timing_score >= threshold

# 会話内容を選択
def select_ability(location, activity):
    df_filtered = df_scores[(df_scores["場所"] == location) & (df_scores["ユーザ活動"] == activity)]
    max_score = df_filtered["選択スコア"].max()
    candidates = df_filtered[df_filtered["選択スコア"] == max_score]
    selected_ability = candidates.sample()["できること"].values[0]
    return selected_ability

# 選択スコアの更新
def update_scores(location, activity, ability, accepted):
    global df_scores
    df_index = (df_scores["場所"] == location) & (df_scores["ユーザ活動"] == activity) & (df_scores["できること"] == ability)
    # スコア更新処理
    if accepted:
        df_scores.loc[df_index, "選択スコア"] += 1  # 受け入れられた場合、スコアを増加
        df_scores.loc[df_index, "受け入れ回数"] += 1
    else:
        df_scores.loc[df_index, "選択スコア"] = max(df_scores.loc[df_index, "選択スコア"].values[0] - 0.5, 0)  # 受け入れられなかった場合、スコアを減少

# メインシミュレーションループ
def simulation_loop():
    initialize_scores()
    while True:
        location = random.choice(locations)
        activity = random.choice(user_activities)
        
        if should_initiate_conversation(location, activity):
            ability = select_ability(location, activity)
            print(f"ロボット: 「{ability}をお手伝いしましょうか？」")
            response = input("ユーザ: (受け入れる: y / 受け入れない: n): ")
            
            accepted = (response.lower() == 'y')
            update_scores(location, activity, ability, accepted)
            
            print("選択スコア更新状況:")
            print(df_scores[df_scores["場所"] == location][df_scores["ユーザ活動"] == activity])
        else:
            print("ロボットは話しかけを控えました。")
        
        cont = input("シミュレーションを続けますか？ (y/n): ")
        if cont.lower() != 'y':
            break

# シミュレーションの実行
simulation_loop()

