import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

from sklearn.linear_model import LinearRegression

warnings.filterwarnings(action='ignore')

League = pd.DataFrame()
League_Object = pd.DataFrame()

# 데이터 가공
def dataProcessing(year_select="2023") :
    global League, League_Object
    if year_select == "2016" :
        League = pd.read_csv('2016_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2017" :
        League = pd.read_csv('2017_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2018" :
        League = pd.read_csv('2018_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2019" :
        League = pd.read_csv('2019_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2020" :
        League = pd.read_csv('2020_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2021" :
        League = pd.read_csv('2021_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2022" :
        League = pd.read_csv('2022_LoL_esports_match_data_from_OraclesElixir.csv')
    else :
        League = pd.read_csv('2023_LoL_esports_match_data_from_OraclesElixir.csv')

    League = League[League['datacompleteness'] == 'complete']
    League = League[League['position'] == 'team']
    League = League[['teamname', 'league', 'result', 'firstdragon', 'firstherald', 'infernals', 'mountains', 'clouds', 'oceans', 'chemtechs', 'hextechs', 'dragons', 'heralds', 'barons']]
    League['dragon_buff'] = (League['dragons'] >= 4.0) * 1
    League['infernal_buff'] = ((League['infernals'] >= 2.0) & League['dragon_buff']) * 1
    League['mountain_buff'] = ((League['mountains'] >= 2.0) & League['dragon_buff']) * 1
    League['cloud_buff'] = ((League['clouds'] >= 2.0) & League['dragon_buff']) * 1
    League['ocean_buff'] = ((League['oceans'] >= 2.0) & League['dragon_buff']) * 1
    League['chemtech_buff'] = ((League['chemtechs'] >= 2.0) & League['dragon_buff']) * 1
    League['hextech_buff'] = ((League['hextechs'] >= 2.0) & League['dragon_buff']) * 1
    League_Object = League.groupby('teamname').agg({'result':'mean'}).sort_values('result')
    League_Object['count'] = League.groupby('teamname').agg({'result':'count'})
    League_Object['firstdragon'] = League.groupby('teamname').agg({'firstdragon':'mean'})
    League_Object['firstherald'] = League.groupby('teamname').agg({'firstherald':'mean'})
    League_Object['dragons'] = League.groupby('teamname').agg({'dragons' : 'mean'})
    League_Object['heralds'] = League.groupby('teamname').agg({'heralds' : 'mean'})
    League_Object['barons'] = League.groupby('teamname').agg({'barons' : 'mean'})
    League_Object['dragon_buff'] = League.groupby('teamname').agg({'dragon_buff' : 'mean'})
    League_Object.drop(League_Object[(League_Object['count'] < 20)].index, inplace=True)

    League_Object['firstdragon_win'] = League.drop(League[(League['firstdragon'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['firstherald_win'] = League.drop(League[(League['firstherald'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['infernal_win'] = League.drop(League[(League['infernal_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['mountain_win'] = League.drop(League[(League['mountain_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['cloud_win'] = League.drop(League[(League['cloud_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['ocean_win'] = League.drop(League[(League['ocean_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['chemtech_win'] = League.drop(League[(League['chemtech_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
    League_Object['hextech_win'] = League.drop(League[(League['hextech_buff'] == 0)].index).groupby('teamname').agg({'result':'mean'})
dataProcessing()

# streamlit 레이아웃 조정
st.set_page_config(layout="wide")
empty1, con1, empty2 = st.columns([0.2, 1.0, 0.2])
empty3, con2, con3, empty4 = st.columns([0.2, 0.5, 0.5, 0.2])
empty5, con4, con5, empty6 = st.columns([0.2, 0.5, 0.5, 0.2])
with con1 :
    st.title("📈오브젝트와 승률의 상관관계 분석")

#streamlit 사이드바
st.sidebar.title('🎮데이터 선택하기')
select_year = st.sidebar.selectbox('분석할 년도를 선택하세요.', ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'])
dataProcessing(select_year)
league_list = np.append(["모든 리그"], League['league'].unique())
select_league = st.sidebar.selectbox('분석할 리그를 선택하세요.', league_list)
if select_league == "모든 리그" :
    team_list = League
else : 
    team_list = League[League['league'] == select_league]
select_team = st.sidebar.selectbox('분석할 팀을 선택하세요.', team_list['teamname'].unique())

def main() :
    if select_team is None :
        st.error("‼️분석할 팀이 없습니다‼️")
        return

    with con2 :
        # 선택한 팀의 첫 오브젝트와 승률 관계 막대 그래프 그리기
        st.header(f"{select_team}팀의 첫 오브젝트와 승률 분석")
        FirstObj_Win = pd.DataFrame({'object':['firstdragon', 'firstherald', 'firstdragon', 'firstherald'],
                                    'type':['average', 'average', 'first_object', 'first_object'],
                                    'win_rate':[League_Object.loc[select_team]['result'], League_Object.loc[select_team]['result'], 
                                                League_Object.loc[select_team]['firstdragon_win'], League_Object.loc[select_team]['firstherald_win']]})
        fig = plt.figure(figsize=(10, 4.7))
        sb.barplot(x='object', y='win_rate', data=FirstObj_Win, hue='type')
        st.pyplot(fig)

        # 그래프 분석
        if League_Object.loc[select_team]['result'] < League_Object.loc[select_team]['firstdragon_win'] :
            st.write(f"- {select_team}팀은 첫 드래곤을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['firstdragon_win'] - League_Object.loc[select_team]['result'])*100:.2f}% 높은 승률을 보여줍니다. 따라서 첫 드래곤을 먹는것이 유리합니다.")    
        else :
            st.write(f"- {select_team}팀은 첫 드래곤을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['result'] - League_Object.loc[select_team]['firstdragon_win'])*100:.2f}% 낮은 승률을 보여줍니다. 따라서 첫 드래곤을 먹는것은 불리합니다.")

        if League_Object.loc[select_team]['result'] < League_Object.loc[select_team]['firstherald_win'] :
            st.write(f"- {select_team}팀은 첫 전령을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['firstherald_win'] - League_Object.loc[select_team]['result'])*100:.2f}% 높은 승률을 보여줍니다. 따라서 첫 전령을 먹는것이 유리합니다.")    
        else :
            st.write(f"- {select_team}팀은 첫 전령을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['result'] - League_Object.loc[select_team]['firstherald_win'])*100:.2f}% 낮은 승률을 보여줍니다. 따라서 첫 전령을 먹는것은 불리합니다.")

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
        st.markdown('''- 첫 드래곤과 승률, 첫 전령과 승률 사이의 관계를 보면 모두 양의 상관관계가 있는 것으로 보여집니다.  
                    붉은색 회귀선이 가리키는 바와 같이, 첫 오브젝트를 더 자주 획득하는 팀이 높은 승률을 보이는 경향이 있습니다.''')
        if lr_dragon_model.coef_[0] > lr_herald_model.coef_[0] :
            st.write(f"- 첫 드래곤의 회귀 계수는 {lr_dragon_model.coef_[0]:.3f}로 첫 전령의 회귀 계수 {lr_herald_model.coef_[0]:.3f}보다 크다. 이를 통해 첫 드래곤을 획득하는 것이 승률에 더 큰 영향을 미친다는 것을 알 수 있다.")
        else :
            st.write(f"- 첫 드래곤의 회귀 계수는 {lr_dragon_model.coef_[0]:.3f}로 첫 전령의 회귀 계수 {lr_herald_model.coef_[0]:.3f}보다 작다. 이를 통해 첫 전령을 획득하는 것이 승률에 더 큰 영향을 미친다는 것을 알 수 있다.")

    with con4 :
        # 선택한 년도의 드래곤 버프 획득과 승률 그래프 그리기
        if int(select_year) < 2020 :
            st.error("드래곤 영혼 출시 이전입니다.")
        else :
            st.header(f"{select_year}년도의 드래곤 영혼 획득과 승률 분석")
            fig = sb.lmplot(x='dragon_buff', y='result', data=League_Object, height=4, line_kws={'color' : 'red'})
            st.pyplot(fig)
            st.markdown('''드래곤 영혼과 승률 사이의 관계를 보면 양의 상관관계가 있는 것으로 보여집니다.  
                        붉은색 회귀선이 가리키는 바와 같이, 드래곤 영혼을 더 자주 획득하는 팀이 높은 승률을 보이는 경향이 있습니다.''')
    
    with con5 :
        if int(select_year) < 2020 :
            st.error("드래곤 영혼 출시 이전입니다.")
        else :
            st.header(f"{select_team}팀의 첫 오브젝트와 승률 분석")
            FirstObj_Win = pd.DataFrame({'object':['infernal', 'mountain', 'cloud', 'ocean', 'chemtech', 'hextech', 'infernal', 'mountain', 'cloud', 'ocean', 'chemtech', 'hextech'],
                                        'type':['average', 'average', 'average', 'average', 'average', 'average', select_team, select_team, select_team, select_team, select_team, select_team],
                                        'win_rate':[League_Object.loc[select_team]['result'], League_Object.loc[select_team]['result'], League_Object.loc[select_team]['result'], League_Object.loc[select_team]['result'], League_Object.loc[select_team]['result'], League_Object.loc[select_team]['result'], 
                                                    League_Object.loc[select_team]['infernal_win'], League_Object.loc[select_team]['mountain_win'], League_Object.loc[select_team]['cloud_win'], League_Object.loc[select_team]['ocean_win'], League_Object.loc[select_team]['chemtech_win'], League_Object.loc[select_team]['hextech_win']]})
            fig = plt.figure(figsize=(10, 4.7))
            sb.barplot(x='object', y='win_rate', data=FirstObj_Win, hue='type')
            st.pyplot(fig)
            win_rate_list = [League_Object.loc[select_team]['infernal_win'], League_Object.loc[select_team]['mountain_win'], League_Object.loc[select_team]['cloud_win'], League_Object.loc[select_team]['ocean_win'], League_Object.loc[select_team]['chemtech_win'], League_Object.loc[select_team]['hextech_win']]
            buff = ['화염', '대지', '바람', '바다', '화학공학', '마법공학']
            st.write(f"{buff(win_rate_list.index(max(win_rate_list)))}의 영혼을 얻었을 때의 승률이 가장 높은것으로 보여집니다.")

main()