from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import numpy as np
import random

# ロボットができることリスト
ABILITIES = ["戸締りを確認", "人を呼ぶ", "雑談を行う", "忘れ物を確認", "書類を整理", "無くしものを探す",\
             "料理レシピを提案", "ペットケアのリマインド", "会議の時間を確認", "家事のリマインド"]

# BERTモデルとトークナイザーのロード（日本語用モデルを指定）
model_name = "cl-tohoku/bert-base-japanese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# 文章のベクトルを取得する関数
def get_ability_vector_BERT(ability):
    inputs = tokenizer(ability, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 文全体のベクトルとして、[CLS]トークンの出力を利用
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return sentence_embedding

# 各abilityのベクトルを辞書に格納
ABILITIES_VECTORS = {ability: get_ability_vector_BERT(ability) for ability in ABILITIES}

# 他のabilitiesに対する類似度計算
SIMILARITIES = {} # -> dict{dict}
for select_ability in ABILITIES:
    temp_similarities = {}
    select_vector = ABILITIES_VECTORS.get(select_ability)
    for ability, vector in ABILITIES_VECTORS.items():
        if ability != select_ability and vector is not None:  # 自身との類似度はスキップ
            similarity = cosine_similarity([select_vector], [vector])[0][0]
            temp_similarities[ability] = similarity
    SIMILARITIES[select_ability] = temp_similarities

print(SIMILARITIES.keys())
print(SIMILARITIES.values())

# 会話内容の選択スコアと実行受け入れ回数を保管するDataFrame
COLUMNS = ["場所", "ユーザ活動", "できること", "選択スコア", "受け入れ回数"]
DF_SCORES = pd.DataFrame(columns=COLUMNS)

# できることの使用回数を管理する辞書
USAGE_COUNTS = {ability: 0 for ability in ABILITIES}

# 場所とユーザ活動のリスト
LOCATIONS = ["bath_room", "bed_room", "living", "kitchen", "landly"]
USER_ACTIVITIES = ["use_internet", "return_from_others", "rest_relax", "cleaning"]

# 初期化関数
def initialize_scores():
    global DF_SCORES
    rows = []
    for location in LOCATIONS:
        for activity in USER_ACTIVITIES:
            for ability in ABILITIES:
                rows.append({
                    "場所": location,
                    "ユーザ活動": activity,
                    "できること": ability,
                    "選択スコア": 1,  # 初期スコアを1に設定
                    "受け入れ回数": 0
                })
    DF_SCORES = pd.concat([DF_SCORES, pd.DataFrame(rows)], ignore_index=True)

# 話しかけるかどうかの判断
def should_initiate_conversation(location, activity, threshold=0.5):
    # TODO: タイミングスコアの計算
    timing_score = random.uniform(0, 1)  # タイミングスコアをランダムに生成
    print("timeing_score: ", timing_score)
    return timing_score >= threshold

# 会話内容を選択
def select_ability(location, activity):
    df_filtered = DF_SCORES[(DF_SCORES["場所"] == location) & (DF_SCORES["ユーザ活動"] == activity)]
    max_score = df_filtered["選択スコア"].max()
    candidates = df_filtered[df_filtered["選択スコア"] == max_score]
    selected_ability = candidates.sample()["できること"].values[0]
    return selected_ability

# 選択スコアの更新(template)
def update_scores_template(location, activity, ability, accepted):
    global DF_SCORES
    df_index = (DF_SCORES["場所"] == location) & (DF_SCORES["ユーザ活動"] == activity) & (DF_SCORES["できること"] == ability)
    USAGE_COUNTS[ability] += 1  # 使用回数をカウントアップ
    # スコア更新処理
    if accepted:
        DF_SCORES.loc[df_index, "選択スコア"] += 1  # 受け入れられた場合、スコアを増加
        DF_SCORES.loc[df_index, "受け入れ回数"] += 1
    else:
        DF_SCORES.loc[df_index, "選択スコア"] = max(DF_SCORES.loc[df_index, "選択スコア"].values[0] - 0.5, 0)  # 受け入れられなかった場合、スコアを減少

# スコア更新方法の実装(BERT)
def update_scores_BERT(location, activity, selected_ability, accepted, importance=0.5):
    global DF_SCORES
    selected_vector = ABILITIES_VECTORS.get(selected_ability)
    
    if selected_vector is None:
        print(f"{selected_ability} のベクトルが存在しないため、スコア更新をスキップします。")
        return

    # DataFrameの更新対象を取得
    df_index = (DF_SCORES["場所"] == location) & (DF_SCORES["ユーザ活動"] == activity) & (DF_SCORES["できること"] == selected_ability)
    USAGE_COUNTS[ability] += 1  # 使用回数をカウントアップ
    
    if accepted:
        # 受け入れられた場合、選択スコアの加重平均を計算しスコアを増加
        for ability, similarity in SIMILARITIES.get(selected_ability).items():
            related_index = (DF_SCORES["場所"] == location) & (DF_SCORES["ユーザ活動"] == activity) & (DF_SCORES["できること"] == ability)
            weighted_increase = importance * similarity  # 類似度に基づく増加量
            DF_SCORES.loc[related_index, "選択スコア"] += weighted_increase
        # 選択されたabilityのスコアも増加
        DF_SCORES.loc[df_index, "選択スコア"] += 1
        DF_SCORES.loc[df_index, "受け入れ回数"] += 1
    else:
        # 受け入れられなかった場合、何もしない
        print("ユーザに受け入れられなかった...")

    print(f"更新されたスコア (場所: {location}, 活動: {activity}):")
    print(DF_SCORES[(DF_SCORES["場所"] == location) & (DF_SCORES["ユーザ活動"] == activity)])

#####

# メインシミュレーションループ
def simulation_loop():
    initialize_scores()
    fin_count = 0
    while True:
        location = random.choice(LOCATIONS)
        activity = random.choice(USER_ACTIVITIES)
        
        if should_initiate_conversation(location, activity):
            ability = select_ability(location, activity)
            #TODO: 定型文
            print(f"ロボット: 「{ability}をお手伝いしましょうか？」")
            response = input("ユーザ: (受け入れる: y / 受け入れない: n): ")
            
            accepted = (response.lower() == 'y')
            #update_scores_template(location, activity, ability, accepted)
            update_scores_BERT(location, activity, ability, accepted, 0.5)
        else:
            print("ロボットは話しかけを控えました。")
        
        fin_count += 1
        if fin_count == 10:
            cont = input("シミュレーションを続けますか？ (y/n): ")
            if cont.lower() != 'y':
                break
            else:
                fin_count = 0

# シミュレーションの実行
simulation_loop()

