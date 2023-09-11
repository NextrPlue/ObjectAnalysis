import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings(action='ignore')

League = pd.DataFrame()
League_Object = pd.DataFrame()

# 데이터 가공
def selectYear(year_select="2023") :
    global League
    if year_select == "2014" :
        League = pd.read_csv('2014_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2015" :
        League = pd.read_csv('2015_LoL_esports_match_data_from_OraclesElixir.csv')
    elif year_select == "2016" :
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

def dataProcessing(league_select="모든 리그", checkbox_btn=True) :
    global League, League_Object
    League = League[League['datacompleteness'] == 'complete']
    League = League[League['position'] == 'team']
    League = League[['teamname', 'league', 'result', 'firstdragon', 'firstherald', 'dragons', 'heralds', 'barons']]
    League['dragon_buff'] = (League['dragons'] >= 4.0) * 1
    if checkbox_btn :
        League = League[League['league'] == league_select]
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
selectYear()
dataProcessing()

# streamlit 레이아웃 조정
st.set_page_config(layout="wide")
empty1, con1, empty2 = st.columns([0.2, 1.0, 0.2])
empty3, con2, con3, empty4 = st.columns([0.2, 0.5, 0.5, 0.2])
with con1 :
    st.title("📈오브젝트와 승률의 상관관계 분석")

#streamlit 사이드바
st.sidebar.title('🎮데이터 선택하기')
select_year = st.sidebar.selectbox('분석할 년도를 선택하세요.', ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'])
selectYear(select_year)
checkbox_btn = st.sidebar.checkbox('선택한 리그 내에서 분석하기')
if checkbox_btn :
    league_list = League['league'].unique()
else :
    league_list = np.append(["모든 리그"], League['league'].unique())
select_league = st.sidebar.selectbox('분석할 리그를 선택하세요.', league_list)
dataProcessing(select_league, checkbox_btn)
if select_league == "모든 리그" :
    team_list = League
else : 
    team_list = League[League['league'] == select_league]
select_team = st.sidebar.selectbox('분석할 팀을 선택하세요.', team_list['teamname'].unique())

# 첫 오브젝트 산점도 그리는 함수
def lmPlot(obj):
    fig = sb.lmplot(x=obj, y="result", data=League_Object, line_kws={'color' : 'red'})
    highlight_x = League_Object.loc[select_team, obj]
    highlight_y = League_Object.loc[select_team, 'result']
    plt.scatter([highlight_x], [highlight_y], color='green')
    plt.annotate(select_team, (highlight_x, highlight_y), textcoords="offset points", xytext=(0,10), ha='center')
    st.pyplot(fig)

def main() :
    if select_team is None :
        st.error("‼️분석할 팀이 없습니다‼️")
        return

    with con2 :
        # 선택한 팀의 첫 오브젝트와 승률 관계 막대 그래프 그리기
        st.header(f"{select_team}팀의 첫 오브젝트와 승률")
        FirstObj_Win = pd.DataFrame({'object':['firstdragon', 'firstherald', 'firstdragon', 'firstherald'],
                                    'type':['average', 'average', 'first_object', 'first_object'],
                                    'win_rate':[League_Object.loc[select_team]['result'], League_Object.loc[select_team]['result'], 
                                                League_Object.loc[select_team]['firstdragon_win'], League_Object.loc[select_team]['firstherald_win']]})
        fig = plt.figure(figsize=(10, 4))
        sb.barplot(x='object', y='win_rate', data=FirstObj_Win, hue='type')
        st.pyplot(fig)

        # 그래프 분석
        if League_Object.loc[select_team]['result'] < League_Object.loc[select_team]['firstdragon_win'] :
            st.write(f"- {select_team}팀은 첫 용을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['firstdragon_win'] - League_Object.loc[select_team]['result'])*100:.2f}% 높은 승률을 보여줍니다. 따라서 첫 용을 먹는것이 유리합니다.")    
        else :
            st.write(f"- {select_team}팀은 첫 용을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['result'] - League_Object.loc[select_team]['firstdragon_win'])*100:.2f}% 낮은 승률을 보여줍니다. 따라서 첫 용을 먹는것은 불리합니다.")

        if League_Object.loc[select_team]['result'] < League_Object.loc[select_team]['firstherald_win'] :
            st.write(f"- {select_team}팀은 첫 전령을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['firstherald_win'] - League_Object.loc[select_team]['result'])*100:.2f}% 높은 승률을 보여줍니다. 따라서 첫 전령을 먹는것이 유리합니다.")    
        else :
            st.write(f"- {select_team}팀은 첫 전령을 먹었을 경우, 평균보다 약 {(League_Object.loc[select_team]['result'] - League_Object.loc[select_team]['firstherald_win'])*100:.2f}% 낮은 승률을 보여줍니다. 따라서 첫 전령을 먹는것은 불리합니다.")

        if League_Object.loc[select_team]['firstdragon_win'] > League_Object.loc[select_team]['firstherald_win'] :
            st.write(f"- 첫 오브젝트로 용을 먹었을 경우의 승률이 전령을 먹었을 때보다 약 {(League_Object.loc[select_team]['firstdragon_win'] - League_Object.loc[select_team]['firstherald_win'])*100:.2f}% 높으므로 전령보단 용을 먹는것이 더 유리합니다.")
        else :
            st.write(f"- 첫 오브젝트로 전령을 먹었을 경우의 승률이 용을 먹었을 때보다 약 {(League_Object.loc[select_team]['firstherald_win'] - League_Object.loc[select_team]['firstdragon_win'])*100:.2f}% 높으므로 용보단 전령을 먹는것이 더 유리합니다.")

    with con3 :
        # 첫 용과 승률 산점도 그래프 그리기
        lmPlot('firstdragon')

        # 첫 전령과 승률 산점도 그래프 그리기
        lmPlot('firstherald')

        #첫 오브젝트와 승률 산점도 그래프 그리기
        df_long = pd.melt(League_Object, id_vars=['result'], value_vars=['firstdragon', 'firstherald'], 
                        var_name='Variable', value_name='Value')
        fig = sb.lmplot(x='Value', y='result', hue='Variable', data=df_long, height=8, aspect=1.2)
        plt.title('Linear Relationship between firstdragon, firstherald and result')

        FirstObj_Win = pd.DataFrame({
            'type': ['firstdragon', 'firstherald'],
            'win_rate': [
                np.average(League_Object['firstdragon_win']),
                np.average(League_Object['firstherald_win'])
            ]
        })
        st.pyplot(fig)

main()