import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings(action='ignore')

League = pd.DataFrame()
League_Object = pd.DataFrame()
League_Predict = pd.DataFrame()
gradient_boosting_model = ""

features = [
    'firstdragon', 'firstherald', 'infernals', 'mountains', 'clouds', 'oceans', 
    'chemtechs', 'hextechs', 'dragons', 'heralds', 'firsttower', 'dragon_buff', 
    'infernal_buff', 'mountain_buff', 'cloud_buff', 'ocean_buff', 'chemtech_buff', 
    'hextech_buff', 'herald_firsttower'
]
target = 'result'

# 데이터 가공
def dataProcessing(year_select="2023") :
    global League, League_Object, League_Predict, gradient_boosting_model
    League = pd.read_csv(f"{year_select}_LoL_esports_match_data_from_OraclesElixir.csv")

    League = League[League['datacompleteness'] == 'complete']
    League = League[League['position'] == 'team']
    League = League[['teamname', 'league', 'result', 'firstdragon', 'firstherald', 'infernals', 'mountains', 'clouds', 'oceans', 'chemtechs', 'hextechs', 'dragons', 'heralds', 'firsttower']]
    League['dragon_buff'] = (League['dragons'] >= 4.0) * 1
    League['infernal_buff'] = ((League['infernals'] >= 2.0) & League['dragon_buff']) * 1
    League['mountain_buff'] = ((League['mountains'] >= 2.0) & League['dragon_buff']) * 1
    League['cloud_buff'] = ((League['clouds'] >= 2.0) & League['dragon_buff']) * 1
    League['ocean_buff'] = ((League['oceans'] >= 2.0) & League['dragon_buff']) * 1
    League['chemtech_buff'] = ((League['chemtechs'] >= 2.0) & League['dragon_buff']) * 1
    League['hextech_buff'] = ((League['hextechs'] >= 2.0) & League['dragon_buff']) * 1
    League['herald_firsttower'] = ((League['heralds'] > 0) & League['firsttower']) * 1

    League_Object = League.groupby('teamname').agg({'result':'mean'}).sort_values('result')
    League_Object['count'] = League.groupby('teamname').agg({'result':'count'})
    League_Object['firstdragon'] = League.groupby('teamname').agg({'firstdragon':'mean'})
    League_Object['firstherald'] = League.groupby('teamname').agg({'firstherald':'mean'})
    League_Object['dragons'] = League.groupby('teamname').agg({'dragons' : 'mean'})
    League_Object['heralds'] = League.groupby('teamname').agg({'heralds' : 'mean'})
    League_Object['dragon_buff'] = League.groupby('teamname').agg({'dragon_buff' : 'mean'})
    League_Object['infernal_count'] = League.groupby('teamname').agg({'infernal_buff' : 'sum'})
    League_Object['mountain_count'] = League.groupby('teamname').agg({'mountain_buff' : 'sum'})
    League_Object['cloud_count'] = League.groupby('teamname').agg({'cloud_buff' : 'sum'})
    League_Object['ocean_count'] = League.groupby('teamname').agg({'ocean_buff' : 'sum'})
    League_Object['chemtech_count'] = League.groupby('teamname').agg({'chemtech_buff' : 'sum'})
    League_Object['hextech_count'] = League.groupby('teamname').agg({'hextech_buff' : 'sum'})
    League_Object['herald_firsttower'] = League.groupby('teamname').agg({'herald_firsttower':'mean'})

    League_Object['firstdragon_win'] = League.drop(League[(League['firstdragon'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['firstherald_win'] = League.drop(League[(League['firstherald'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['infernal_win'] = League.drop(League[(League['infernal_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['mountain_win'] = League.drop(League[(League['mountain_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['cloud_win'] = League.drop(League[(League['cloud_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['ocean_win'] = League.drop(League[(League['ocean_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['chemtech_win'] = League.drop(League[(League['chemtech_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['hextech_win'] = League.drop(League[(League['hextech_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['herald_firsttower_win'] = League.drop(League[(League['herald_firsttower'] == 0)].index).groupby('teamname').agg({'result':'mean'})

    League_Predict = League
    League_Predict['chemtechs'].fillna(0, inplace=True)
    League_Predict['hextechs'].fillna(0, inplace=True)
    League_Predict['infernals'].fillna(0, inplace=True)
    League_Predict['mountains'].fillna(0, inplace=True)
    League_Predict['clouds'].fillna(0, inplace=True)
    League_Predict['oceans'].fillna(0, inplace=True)
    League_Predict['dragons'].fillna(0, inplace=True)

    columns_to_fill_mean = ['firstdragon', 'heralds', 'firsttower', 'firstherald']
    for column in columns_to_fill_mean:
        mean_value = League_Predict[column].mean()
        League_Predict[column].fillna(mean_value, inplace=True)

# 승부 예측 함수
def predictWinner(team1, team2) :
    # Split the League_Predict into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(League_Predict[features], League_Predict[target], test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train the model
    gradient_boosting_model.fit(X_train, y_train)

    # Predict the outcomes for the test set
    y_pred = gradient_boosting_model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Filter the data to include only the matches involving the specified teams
    team1_data = League[League['teamname'] == team1]
    team2_data = League[League['teamname'] == team2]

    # Calculate the mean statistics for each team
    team1_mean_stats = team1_data[features].mean()
    team2_mean_stats = team2_data[features].mean()

    # Reshape the data to match the model's input shape
    team1_mean_stats = team1_mean_stats.values.reshape(1, -1)
    team2_mean_stats = team2_mean_stats.values.reshape(1, -1)

    # Use the best model to predict the win probability for each team
    team1_win_prob = gradient_boosting_model.predict_proba(team1_mean_stats)[:, 1]
    team2_win_prob = gradient_boosting_model.predict_proba(team2_mean_stats)[:, 1]

    # Calculate the normalized win probabilities for each team
    total_prob = team1_win_prob + team2_win_prob
    normalized_team1_win_prob = (team1_win_prob / total_prob) * 100
    normalized_team2_win_prob = (team2_win_prob / total_prob) * 100

    total_prob = team1_win_prob + team2_win_prob
    normalized_team1_win_prob = (team1_win_prob / total_prob) * 100
    normalized_team2_win_prob = (team2_win_prob / total_prob) * 100

    print(normalized_team1_win_prob)
    print(normalized_team2_win_prob)

    return normalized_team1_win_prob, normalized_team2_win_prob, accuracy

# streamlit 레이아웃 조정
st.set_page_config(layout="wide")
empty1, con1, empty2 = st.columns([0.1, 1.0, 0.1])
empty3, con2, con3, empty4 = st.columns([0.1, 0.5, 0.5, 0.1])
empty5, con4, con5, empty6 = st.columns([0.1, 0.5, 0.5, 0.1])
empty7, con6, con7, empty8 = st.columns([0.1, 0.5, 0.5, 0.1])
empty9, con8, empty9 = st.columns([0.2, 1.0, 0.2])
with con1 :
    st.title("📈오브젝트와 승률의 상관관계 분석")

#streamlit 사이드바
st.sidebar.title('🎮데이터 선택하기')
select_year = st.sidebar.selectbox('분석할 년도를 선택하세요.', ['2018', '2019', '2020', '2021', '2022', '2023'])
select_year = "2022"
dataProcessing(select_year)
league_list = np.append(["모든 리그"], sorted(League['league'].unique()))
select_league = st.sidebar.selectbox('분석할 리그를 선택하세요.', league_list)
team_list = League[['teamname', 'league']]
if select_league != "모든 리그" :
    team_list = team_list[team_list['league'] == select_league]
select_team = st.sidebar.selectbox('분석할 팀을 선택하세요.', sorted(team_list['teamname'].unique().astype(str)))
min_match = st.sidebar.slider('필요한 최소 경기 수를 선택하세요.', 10, 50, 20, 5)

def main() :
    if select_team is None :
        st.error("‼️분석할 팀이 없습니다‼️")
        return
    if League_Object.loc[select_team]['count'] < min_match :
        st.error(f"매치 수가 {min_match}회 미만입니다.")
        return
    
    League_Object.drop(League_Object[(League_Object['count'] < min_match)].index, inplace=True)
    st.sidebar.dataframe(League_Object.loc[select_team])
    
    with con2 :
        # 선택한 팀의 첫 오브젝트와 승률 관계 막대 그래프 그리기
        st.header(f"{select_team}팀의 첫 오브젝트와 승률 분석")
        FirstObj_Win = pd.DataFrame({'object':['firstdragon', 'firstherald'],
                                    'win_rate':[League_Object.loc[select_team]['firstdragon_win'], League_Object.loc[select_team]['firstherald_win']]})
        fig = plt.figure(figsize=(10, 4.7))
        ax = fig.add_subplot()
        sb.barplot(x='object', y='win_rate', data=FirstObj_Win, label=select_team)
        ax.axhline(League_Object.loc[select_team]['result'], color='red', linestyle='solid', label='mean')
        ax.legend()
        st.pyplot(fig)

        # 그래프 분석
        if League_Object.loc[select_team]['result'] < League_Object.loc[select_team]['firstdragon_win'] :
            st.write(f"- {select_team}팀은 첫 드래곤을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['firstdragon_win'] - League_Object.loc[select_team]['result'])*100:.2f}% 높은 승률을 보여줍니다. 따라서 첫 드래곤을 먹는것은 유리합니다.")    
        else :
            st.write(f"- {select_team}팀은 첫 드래곤을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['result'] - League_Object.loc[select_team]['firstdragon_win'])*100:.2f}% 낮은 승률을 보여줍니다. 따라서 첫 드래곤을 먹는것은 불리합니다.")

        if League_Object.loc[select_team]['result'] < League_Object.loc[select_team]['firstherald_win'] :
            st.write(f"- {select_team}팀은 첫 전령을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['firstherald_win'] - League_Object.loc[select_team]['result'])*100:.2f}% 높은 승률을 보여줍니다. 따라서 첫 전령을 먹는것은 유리합니다.")    
        else :
            st.write(f"- {select_team}팀은 첫 전령을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['result'] - League_Object.loc[select_team]['firstherald_win'])*100:.2f}% 낮은 승률을 보여줍니다. 따라서 첫 전령을 먹는것은 불리합니다.")
        if (League_Object.loc[select_team]['firstdragon_win'] < League_Object.loc[select_team]['result']) & \
            (League_Object.loc[select_team]['firstherald_win'] < League_Object.loc[select_team]['result']) :
            st.write(f"- 첫 오브젝트를 얻었을 경우 전령과 드래곤 모두 평균보다 낮은 승률을 보여줍니다. 따라서 첫 오브젝트를 챙기는 것은 불리합니다.")
        else :
            if League_Object.loc[select_team]['firstdragon_win'] > League_Object.loc[select_team]['firstherald_win'] :
                st.write(f"- 첫 오브젝트로 드래곤을 먹었을 경우의 승률이 전령을 먹었을 때보다 약 {(League_Object.loc[select_team]['firstdragon_win'] - League_Object.loc[select_team]['firstherald_win'])*100:.2f}% 높으므로 전령보단 드래곤을 먹는것이 더 유리합니다.")
            else :
                st.write(f"- 첫 오브젝트로 전령을 먹었을 경우의 승률이 드래곤을 먹었을 때보다 약 {(League_Object.loc[select_team]['firstherald_win'] - League_Object.loc[select_team]['firstdragon_win'])*100:.2f}% 높으므로 드래곤보단 전령을 먹는것이 더 유리합니다.")
        
    with con3 :
        # 선택한 년도의 첫 오브젝트와 승률 산점도, 회귀선, 신뢰 구간 그래프 그리기
        st.header(f"{select_year}년도의 첫 오브젝트와 승률 분석")
        fig = sb.PairGrid(League_Object, y_vars=["result"], x_vars=["firstdragon", "firstherald"], height=4)
        fig.map(sb.regplot, line_kws={'color' : 'red'})
        st.pyplot(fig)

        # 회귀 계수와 적합도 분석
        Xd = League_Object[['firstdragon']]
        yd = League_Object['result']
        lr_dragon_model = LinearRegression()
        lr_dragon_model.fit(Xd, yd)
        st.write(f"첫 드래곤의 회귀 계수 : {lr_dragon_model.coef_[0]:.3f}, 결정 계수 : {lr_dragon_model.score(Xd, yd):.3f}")

        Xh = League_Object[['firstherald']]
        yh = League_Object['result']
        lr_herald_model = LinearRegression()
        lr_herald_model.fit(Xh, yh)
        st.write(f"첫 전령의 회귀 계수 : {lr_herald_model.coef_[0]:.3f}, 결정 계수 : {lr_herald_model.score(Xh, yh):.3f}")


        # 그래프 분석
        st.write('- 첫 드래곤과 승률, 첫 전령과 승률 사이의 관계를 보면 모두 양의 상관관계가 있는 것으로 보여집니다.')
        st.write('- 붉은색 회귀선이 가리키는 바와 같이, 첫 오브젝트를 더 자주 획득하는 팀이 높은 승률을 보이는 경향이 있습니다.')
        if lr_dragon_model.coef_[0] > lr_herald_model.coef_[0] :
            st.write(f"- 첫 드래곤의 회귀 계수는 {lr_dragon_model.coef_[0]:.3f}로 첫 전령의 회귀 계수 {lr_herald_model.coef_[0]:.3f}보다 큽니다. 이를 통해 첫 드래곤을 획득하는 것이 승률에 더 큰 영향을 미친다는 것을 알 수 있습니다.")
        else :
            st.write(f"- 첫 드래곤의 회귀 계수는 {lr_dragon_model.coef_[0]:.3f}로 첫 전령의 회귀 계수 {lr_herald_model.coef_[0]:.3f}보다 작습니다. 이를 통해 첫 전령을 획득하는 것이 승률에 더 큰 영향을 미친다는 것을 알 수 있습니다.")

    with con4 :
        # 선택한 년도의 드래곤 처치 수와 승률 그래프 그리기
        st.header(f"{select_year}년도의 드래곤 처치 수와 승률 분석")
        fig = sb.lmplot(x='dragons', y='result', data=League_Object, height=4, line_kws={'color' : 'red'})
        st.pyplot(fig)

        # 회귀 계수와 적합도 분석
        Xds = League_Object[['dragons']]
        yds = League_Object['result']
        lr_dragons_model = LinearRegression()
        lr_dragons_model.fit(Xds, yds)
        st.write(f"그래프의 회귀 계수 : {lr_dragons_model.coef_[0]:.3f}, 결정 계수 : {lr_dragons_model.score(Xds, yds):.3f}")

        # 그래프 분석
        st.write('- 드래곤 처치 수와 승률 사이의 관계를 보면 모두 양의 상관관계가 있는 것으로 보여집니다.')
        st.write('- 붉은색 회귀선이 가리키는 바와 같이, 드래곤을 더 많이 획득하는 팀이 높은 승률을 보이는 경향이 있습니다.')
        
    with con5 :
        # 선택한 년도의 첫 전령과 첫 타워, 첫 타워와 승률 그래프 그리기
        st.header(f"{select_year}년도의 전령 버프를 이용한 첫 타워와 승률 분석")
        fig = sb.lmplot(x='herald_firsttower', y='result', data=League_Object, height=4, line_kws={'color' : 'red'})
        st.pyplot(fig)

        # 회귀 계수와 적합도 분석
        Xht = League_Object[['herald_firsttower']]
        yht = League_Object['result']
        lr_heraldt_model = LinearRegression()
        lr_heraldt_model.fit(Xht, yht)
        st.write(f"그래프의 회귀 계수 : {lr_heraldt_model.coef_[0]:.3f}, 결정 계수 : {lr_heraldt_model.score(Xht, yht):.3f}")
        
        # 그래프 분석
        st.write('- 전령 버프를 이용한 첫 타워 파괴와 승률 사이의 관계를 보면 모두 양의 상관관계가 있는 것으로 보여집니다.')
        st.write('- 붉은색 회귀선이 가리키는 바와 같이, 전령 버프를 이용하여 첫 타워를 파괴하는 팀이 높은 승률을 보이는 경향이 있습니다.')

    with con6 :
        # 선택한 팀의 전령 버프를 이용한 첫 타워와 승률 그래프 그리기
        st.header(f"{select_team}팀의 전령 버프를 이용한 첫 타워와 승률 분석")
        Firsttower_Win = pd.DataFrame({'result':['herald_firsttower', 'average'],
                            'win_rate':[League_Object.loc[select_team]['herald_firsttower_win'], League_Object.loc[select_team]['result']]})
        fig = plt.figure(figsize=(10, 4.7))
        ax = fig.add_subplot()
        sb.barplot(x='result', y='win_rate', data=Firsttower_Win)
        st.pyplot(fig)

        # 그래프 분석
        if League_Object.loc[select_team]['herald_firsttower_win'] > League_Object.loc[select_team]['result'] :
            st.write(f"- {select_team}팀은 전령 버프를 이용하여 첫 타워를 부쉈을 경우, 평균보다 약 {(League_Object.loc[select_team]['herald_firsttower_win'] - League_Object.loc[select_team]['result'])*100:.2f} 높은 승률을 보여줍니다.\
                     따라서 전령 버프를 이용하여 첫 타워를 부수는 것이 유리합니다.")
        else :
            st.write(f"- {select_team}팀은 전령 버프를 이용하여 첫 타워를 부쉈을 경우, 평균보다 약 {(League_Object.loc[select_team]['result'] - League_Object.loc[select_team]['herald_firsttower_win'])*100:.2f} 낮은 승률을 보여줍니다.\
                     따라서 전령 버프를 이용하여 첫 타워를 부수는 것은 불리합니다.")

        # 선택한 팀의 드래곤 버프 획득과 승률 그래프 그리기
        st.header(f"{select_team}팀의 드래곤 영혼과 승률 분석")
        if int(select_year) < 2020 :
            st.error("드래곤 영혼 출시 이전입니다.")
        else :
            buff_eng = ['infernal', 'mountain', 'cloud', 'ocean', 'chemtech', 'hextech']
            buff_object = [buff_eng[i] for i in range (len(buff_eng)) if League_Object.loc[select_team][(buff_eng[i]+'_count')] != 0]
            
            Buff_Win = pd.DataFrame({'type':buff_object,
                                        'win_rate':[League_Object.loc[select_team][(i+'_win')] for i in buff_object]})
            fig = plt.figure(figsize=(10, 4.7))
            ax = fig.add_subplot()
            sb.barplot(x='type', y='win_rate', data=Buff_Win, label=select_team)
            ax.axhline(League_Object.loc[select_team]['result'], color='red', linestyle='solid', label='mean')
            ax.legend()
            st.pyplot(fig)
            
            # 그래프 분석
            win_rate_list = [League_Object.loc[select_team]['infernal_win'], League_Object.loc[select_team]['mountain_win'], League_Object.loc[select_team]['cloud_win'], League_Object.loc[select_team]['ocean_win'], League_Object.loc[select_team]['chemtech_win'], League_Object.loc[select_team]['hextech_win']]
            buff = ['화염', '대지', '바람', '바다', '화학공학', '마법공학']
            max_buff = []
            low_buff = []
            for i in range (len(win_rate_list)) :
                if win_rate_list[i] == max(win_rate_list) :
                    max_buff.append(buff[i])
            for i in range (len(win_rate_list)) :
                if win_rate_list[i] < League_Object.loc[select_team]['result'] :
                    low_buff.append(buff[i])
            if (max(win_rate_list) >= League_Object.loc[select_team]['result']) :
                st.write(f"- {', '.join(max_buff)}의 영혼을 얻었을 경우, 평균보다 약 {(max(win_rate_list) - League_Object.loc[select_team]['result'])*100:.2f}% 높은 승률을 보여주며 가장 높은 승률을 기록하였습니다. 따라서 {', '.join(max_buff)}의 영혼을 얻는 것이 상대적으로 유리합니다.")
                if (min(win_rate_list) < League_Object.loc[select_team]['result']) :
                    st.write(f"- {', '.join(low_buff)}의 영혼을 얻었을 경우 평균보다 낮은 승률을 보여줍니다. 따라서 {', '.join(low_buff)}의 영혼은 피하는 것이 상대적으로 유리합니다.")
            else :
                st.write(f"- 드래곤의 영혼을 얻었을 경우의 승률이 평균보다 낮습니다. 따라서 드래곤의 영혼을 얻는 것은 불리합니다.")

    with con7 :
        # 선택한 년도의 드래곤 버프 획득과 승률 그래프 그리기
        st.header(f"{select_year}년도의 드래곤 영혼 획득과 승률 분석")
        if int(select_year) < 2020 :
            st.error("드래곤 영혼 출시 이전입니다.")
        else :
            fig = sb.lmplot(x='dragon_buff', y='result', data=League_Object, height=4, line_kws={'color' : 'red'})
            st.pyplot(fig)

            # 회귀 계수와 적합도 분석
            Xdb = League_Object[['dragon_buff']]
            ydb = League_Object['result']
            lr_dragonbuff_model = LinearRegression()
            lr_dragonbuff_model.fit(Xdb, ydb)
            st.write(f"그래프의 회귀 계수 : {lr_dragonbuff_model.coef_[0]:.3f}, 결정 계수 : {lr_dragonbuff_model.score(Xdb, ydb):.3f}")

            # 그래프 분석
            st.write('- 드래곤 영혼과 승률 사이의 관계를 보면 양의 상관관계가 있는 것으로 보여집니다.')
            st.write('- 붉은색 회귀선이 가리키는 바와 같이, 드래곤 영혼을 더 자주 획득하는 팀이 높은 승률을 보이는 경향이 있습니다.')
    
    with con8 :
        st.header(f"{select_team}팀의 승부 예측")
        select_team2 = st.selectbox('대결할 팀을 선택하세요.', sorted(League_Predict['teamname'].unique().astype(str)))
        team1_result, team2_result, accuracy = predictWinner(select_team, select_team2)
        col1, col2, col3 = st.columns(3)
        col1.metric(select_team, team1_result[0], team1_result[0] - team2_result[0])
        col2.metric(select_team2, team2_result[0], team2_result[0] - team1_result[0])
        col3.metric("Accuracy", accuracy)

main()