# from cmath import phase
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from scipy.spatial import ConvexHull
from sqlalchemy import create_engine
import sqlalchemy

## Helpers
routine_conv = {
    '1st': 0,
    '2nd': 1
}
## Functions
@st.cache_resource
def init_engine():
    engine = create_engine(
    sqlalchemy.engine.url.URL.create(
        **st.secrets["mysql"]
    ),
    pool_recycle=3600
    )
    return engine

@st.cache_data(ttl=600)
def make_call(sql_str, _engine):
    with _engine.connect() as con:
        df = pd.read_sql(sql_str, con)
    return df

def calculate_ranks(df, rating):
    df = df.reset_index()
    rating_list = []
    if rating != "HD3 Original":
        entry_full = 'Sum ' + rating
        df[entry_full] = df['Difficulty'] + df['Time of flight'] + df['Execution'] + df[rating] + df['Penalty']
    else:
        entry_full = 'Total'

    rankchange_list = []
    meanchange_list = []
    idx_list = []
    df_routine01 = df[df["Routine"] == "1st"]
    df_routine02 = df[df["Routine"] == "2nd"]
    for idx, row in df_routine01.iterrows():
        if phase == "Qualification": # if qual we need to cope with two routines
            test = df_routine02[df_routine02["Name"] == row["Name"]] # find athletes with two routines
            if len(test) == 0: # if we didn't find a match drop -> only 1 routine
                idx_list.append(idx)
            else:
                if "Europe" in event_str: # europe events only count better routine
                    if row[entry_full] < test[entry_full].iloc[0]: # if the other routine was better, change entry
                        df_routine01.loc[idx:idx] = test.values
                else: # non europe
                    # print(f"{row[entry_full]} {test[entry_full]}")
                    if row[entry_full] > test[entry_full].iloc[0]: # if the routin is better then calc sum score
                        df_routine01.at[idx, entry_full] = row[entry_full] + test[entry_full].iloc[0]
                    else: # if other routine is better then change routines and calc sum score
                        # st.write(test.values)
                        df_routine01.loc[idx:idx] = test.values
                        df_routine01.at[idx, entry_full] = row[entry_full] + test[entry_full].iloc[0]
        else:
            if rating != "HD3 Original":
                if row[rating] == 0: # remove 0 ratings
                    idx_list.append(idx)



    # for idx2, row2 in df.iterrows():
    #     if idx2 > 0:
    #         if (rating != "HD3 Original"):
    #             if (row2[rating] == 0.0):
    #             #     # print("Dropped because no file")
    #                 idx_list.append(idx2)
    #             #     if (row2["Name"] == df.iloc[idx2-1]["Name"]) & (row2["Routine"] == "2nd"):
    #             #         idx_list.append(idx2-1)
    #         else:
    #             if phase == "Qualification":
    #                 if row2["Name"] == df.iloc[idx2-1]["Name"]:
    #                     if "Europe" in event_str:
    #                         if (row2[entry_full] > df.iloc[idx2-1][entry_full]):
    #                             idx_list.append(idx2-1)
    #                         else:
    #                             idx_list.append(idx2)
    #                     else:
    #                         df.at[idx2, entry_full] = row2[entry_full] + df.at[idx2-1, entry_full]
    #                         idx_list.append(idx2-1)

    # st.write(df)
    df_result = df_routine01.drop(idx_list, axis=0)
    df_result = df_result.sort_values(by=[entry_full], ascending=False)
    new_rank = df_result["Rank"]
    df_result['New Rank'] = np.sort(new_rank)
    rank_change_unsigned = df_result['New Rank']-df_result['Rank']
    rank_change_unsigned_str = [str(-i) if i >= 0 else "+"+str(-i) for i in rank_change_unsigned]
    rank_change = np.abs(rank_change_unsigned)
    rankchange_list.append(np.sum(rank_change!=0))
    meanchange_list.append(np.mean(rank_change))
    df_result["Rankchange"] = rank_change_unsigned_str

    pop_str = ["HD3 Distance", "HD5 Distance", "HD3 Error", "HD5 Error", "HD3 Hull", "HD5 Hull", "Event", "Phase", "Year", "Location", "Year", "index", "Gender", "Routine", "Rank", "Qualified", "Hash", "level_0"]
    if rating != "HD3 Original":
        pop_str.remove(rating)
    df_result = df_result.drop(pop_str, axis=1)
    return df_result

st.set_page_config(page_title="Trampoline Dashboard",
                   page_icon=":running:",
                   layout="wide")


# Defaults
hash_val = 'empty'
exercise = 'All'
event_str = 'All'
gender = 'All'
athlete = 'All'
rating_str = ("HD3 Original", "HD3 Distance", "HD5 Distance", "HD3 Error", "HD5 Error", "HD3 Hull", "HD5 Hull")

engine = init_engine()
st.markdown('# Trampoline Data App')
sql_str = "SELECT * from ranklists"
df = make_call(sql_str, engine)
df['Event Name'] = df["Year"].astype(str) + " " + df["Event"]

## Sidebar
st.sidebar.markdown('## Info')
st.sidebar.markdown('Go to [readme](https://github.com/falkoin/portfolio#readme) to get more information.')
st.sidebar.markdown('[Repository](https://github.com/falkoin/portfolio) with source code.')
st.sidebar.markdown('## Explore Data')
st.sidebar.markdown('*Make selections to explore data*')
show_dataframe = st.sidebar.checkbox('Show data', value=True)
if show_dataframe:
    st.markdown("### Selected data")

# Event

event_str = st.sidebar.selectbox(
    'Select Event:',
    (['All'] + df["Event Name"].unique().tolist())
    )

df_temp = df[df['Event Name']==event_str]
df_temp.reset_index(inplace=True)
# st.write(df_temp["Year"])
sql_str = ""

if event_str == 'All':
    sql_str = "SELECT * from ranklists"
else:
    sql_str = "SELECT * from ranklists where event=" + "'" + df_temp["Event"][0] + "' and year=" + "'" + df_temp["Year"][0].astype(str) + "'"
    # st.write(sql_str)
df_select = make_call(sql_str, engine)

# Gender

gender = st.sidebar.selectbox(
     'Select Gender:',
     (['All'] + df_select["Gender"].unique().tolist())
    )

if gender == 'All':
    sql_str = "SELECT * from " + "(" + sql_str + ") AS T "
else:
    sql_str = "SELECT * from " + "(" + sql_str + ") AS T " + "where gender=" + "'" + gender + "'"
    # st.write(sql_str)


df_select = make_call(sql_str, engine)

# Athlte

athlete = st.sidebar.selectbox(
     'Select Athlete:',
     (['All'] + df_select["Name"].unique().tolist())
    )

if athlete == 'All':
    sql_str = "SELECT * from " + "(" + sql_str + ") AS T "
else:
    sql_str = "SELECT * from " + "(" + sql_str + ") AS T " + "where name=" + "'" + athlete + "'"


df_select = make_call(sql_str, engine)

# df_select.droplevel("index")
df_select.drop(['index'], axis=1, inplace=True)
if show_dataframe:
    st.dataframe(df_select.drop(["Hash", "HD3 Distance", "HD5 Distance", "HD3 Error", "HD5 Error", "HD3 Hull", "HD5 Hull"], axis=1))
if athlete != 'All':
    exercise = st.sidebar.selectbox(
        'Select Routine:',
        (['All'] + [str(i) for i in np.arange(1, len(df_select)+1)] )
        )
if exercise != 'All':
    exercise = str(int(exercise)-1)
# debug = st.sidebar.checkbox('Debug')
debug = False
if exercise != 'All':
    hash_val = df_select["Hash"].iloc[int(exercise)]
    # st.write(hash_val)
    if hash_val != '':
        sql_str2 = "SELECT * from `routinedata` where hash= '" + hash_val + "'"

        df_exercisedata = make_call(sql_str2, engine)
        df_exercisedata = df_exercisedata.drop('Hash', axis=1)
        df_exercisedata = df_exercisedata.astype(float)
        x = df_exercisedata[['x']].values
        y = df_exercisedata[['y']].values
        # hull
        hd_hull3 = df_select["HD3 Hull"].iloc[int(exercise)]
        hd_hull5 = df_select["HD5 Hull"].iloc[int(exercise)]
        # distance
        hd_distance3 = df_select["HD3 Distance"].iloc[int(exercise)]
        hd_distance5 = df_select["HD5 Distance"].iloc[int(exercise)]
        # error
        hd_error3 = df_select["HD3 Error"].iloc[int(exercise)]
        hd_error5 = df_select["HD5 Error"].iloc[int(exercise)]


# st.write(df_exercisedata)


left_column, right_column = st.columns(2)


with left_column:
    if len(athlete) > 3:
        df_polar = pd.melt(df_select, id_vars=['Rank', 'Name','Event', 'Routine', 'Country', 'Penalty',
                                               'Total', 'End', 'Phase', 'Qualified', 'Location', 'Year',
                                               'Gender', 'Hash',"HD3 Distance", "HD5 Distance", "HD3 Error",
                                               "HD5 Error", "HD3 Hull", "HD5 Hull"], var_name='Rating').sort_values(['Rank', 'Name'])
        df_polar['Routines'] = df_polar["Year"].astype(str) + " " + df_polar["Event"] + " " + df_polar["Phase"] + " " + df_polar["Routine"]

        fig = px.line_polar(

            df_polar,
            title=athlete,
            r="value",
            theta="Rating",
            line_close=True,
            color="Routines",
            range_r = [0, 20]

        )
        st.plotly_chart(fig)
        # st.write(df_exercisedata)
        if hash_val != '':
            if exercise != 'All':
                bar_text = df_exercisedata["index"]+1
                fig3 = px.bar(
                    df_exercisedata,
                    x=bar_text,
                    y='T',
                    title=f"Time of Flight: {np.round(df_exercisedata['T'].sum(), 2)}"
                    )
                fig3.update_xaxes(title_text='Jumps')
                fig3.update_yaxes(title_text='Time (s)')
                st.plotly_chart(fig3)

trampoline_list = [
                    [[54, 54, -54, -54, 54], [54, -54, -54, 54, 54]],
                    [[214, -214], [54, 54]],
                    [[214, -214], [-54, -54]],
                    [[-107.5, -107.5], [-107, 107]],
                    [[107.5, 107.5], [-107, 107]],
                    [[-35, 35], [0, 0]],
                    [[0, 0], [-35, 35]],
                ]
with right_column:
    if hash_val != '':
        if exercise != 'All':
            # st.write('HD:')

            picked_rating = st.radio('Pick a rating approach', rating_str, index=0, horizontal=True)
            if 'HD5' in picked_rating:
                lower_limit = 5
                upper_limit = 10
                color_vec = np.repeat([df_select[picked_rating].iloc[int(exercise)]], 10)
                color_title = 'Routine'
            elif picked_rating == 'HD3 Original':
                lower_limit = 0.7
                upper_limit = 1.0
                color_vec = 'H'
                color_title = 'Jump'
            elif 'HD3' in picked_rating:
                lower_limit = 7
                upper_limit = 10
                color_vec = np.repeat([df_select[picked_rating].iloc[int(exercise)]], 10)
                color_title = 'Routine'
            show_visualization = st.checkbox('Visualize approach', value=False)
            if picked_rating == "HD3 Original":
                hd_title = df_select["Horizontal displacement"].loc[int(exercise)]
            elif picked_rating == "HD3 Hull":
                hd_title = hd_hull3
            elif picked_rating == "HD5 Hull":
                hd_title = hd_hull5
            elif picked_rating == "HD3 Distance":
                hd_title = hd_distance3
            elif picked_rating == "HD5 Distance":
                hd_title = hd_distance5
            elif picked_rating == "HD3 Error":
                hd_title = hd_error3
            elif picked_rating == "HD5 Error":
                hd_title = hd_error5
            # )
            scatter_text = df_exercisedata["index"]+1
            fig2 = px.scatter(
                df_exercisedata,
                x='x',
                y='y',
                text=scatter_text,
                color=color_vec,
                range_color=(lower_limit, upper_limit),
                title=f"Horizontal Displacement: {hd_title}"
                )
            fig2.update_traces(textposition='bottom right')
            fig2.update_layout(
                xaxis_range=[-214, 214],
                yaxis_range=[-107, 107],
                xaxis_visible=False,
                xaxis_showticklabels=False,
                yaxis_visible=False,
                yaxis_showticklabels=False,
                coloraxis={"colorscale": [[0, "red"], [0.5, "yellow"], [1, "green"]]},
                coloraxis_colorbar={"title": color_title}
                )
            for entry in trampoline_list:
                fig2.add_trace(
                    px.line(
                        x=entry[0],
                        y=entry[1],
                        color_discrete_sequence=['#7f7f7f']
                    ).data[0]
                )
            if show_visualization and picked_rating == "HD3 Original":
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=30,
                                            y=45,
                                            showarrow=False,
                                            text="0.0",
                                            textangle=0,
                                            xanchor='left'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=83,
                                            y=45,
                                            showarrow=False,
                                            text="0.1",
                                            textangle=0,
                                            xanchor='left'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=190,
                                            y=45,
                                            showarrow=False,
                                            text="0.2",
                                            textangle=0,
                                            xanchor='left'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=190,
                                            y=97,
                                            showarrow=False,
                                            text="0.3",
                                            textangle=0,
                                            xanchor='left'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=10,
                                            y=97,
                                            showarrow=False,
                                            text="0.2",
                                            textangle=0,
                                            xanchor='right'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=-83,
                                            y=45,
                                            showarrow=False,
                                            text="0.1",
                                            textangle=0,
                                            xanchor='right'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=-190,
                                            y=45,
                                            showarrow=False,
                                            text="0.2",
                                            textangle=0,
                                            xanchor='right'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=-190,
                                            y=-63,
                                            showarrow=False,
                                            text="0.3",
                                            textangle=0,
                                            xanchor='right'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=-190,
                                            y=97,
                                            showarrow=False,
                                            text="0.3",
                                            textangle=0,
                                            xanchor='right'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=190,
                                            y=-63,
                                            showarrow=False,
                                            text="0.3",
                                            textangle=0,
                                            xanchor='left'
                                    ))
                fig2.add_annotation(dict(font=dict(color='black',size=15),
                                            x=10,
                                            y=-63,
                                            showarrow=False,
                                            text="0.2",
                                            textangle=0,
                                            xanchor='right'
                                    ))
            elif show_visualization and 'Distance' in picked_rating :
                list1a = df_exercisedata['x'].to_list()
                list1b = df_exercisedata['y'].to_list()
                list2 = [0] * len(list1a)
                resulta = [None] * (len(list1a) + len(list2))
                resultb = [None] * (len(list1a) + len(list2))
                resulta[::2] = list1a
                resulta[1::2] = list2
                resultb[::2] = list1b
                resultb[1::2] = list2
                fig2.add_trace(
                    px.line(
                        x=resulta,
                        y=resultb,
                        color_discrete_sequence=['rgba(150, 150, 150, 0.5)']
                    ).data[0]
                )
            elif show_visualization and 'Hull' in picked_rating:
                list1a = df_exercisedata['x'].to_numpy()
                test_nan = np.sum(np.isnan(list1a))
                valid_values = len(x)-test_nan
                if valid_values > 2:
                    list1b = df_exercisedata['y'].to_numpy()
                    hull_pts = ConvexHull(np.column_stack((list1a,list1b)))
                    fig2.add_trace(
                        px.line(
                            x=np.append(list1a[hull_pts.vertices], list1a[hull_pts.vertices[0]]),
                            y=np.append(list1b[hull_pts.vertices], list1b[hull_pts.vertices[0]]),
                            color_discrete_sequence=['rgba(150, 150, 150, 0.5)']
                        ).data[0]
                    )
            elif show_visualization and 'Error' in picked_rating :
                list1a = df_exercisedata['x'].to_list()
                list1b = df_exercisedata['y'].to_list()
                list1a = [i*2 if abs(i) > 107.5 else i for i in list1a] # account for values in more dangerous parts of the trampoline
                list1b = [i*2 if abs(i) > 54 else i for i in list1b]
                list1b = [i * 2 for i in list1b]
                list2 = [0] * len(list1a)
                resulta = [None] * (len(list1a) + len(list2))
                resultb = [None] * (len(list1a) + len(list2))
                resulta[::2] = list1a
                resulta[1::2] = list2
                resultb[::2] = list1b
                resultb[1::2] = list2
                fig2.add_trace(
                    px.line(
                        x=resulta,
                        y=resultb,
                        color_discrete_sequence=['rgba(150, 150, 150, 0.5)']
                    ).data[0],
                )
                fig2.add_trace(
                    px.scatter(
                    df_exercisedata,
                    x=list1a,
                    y=list1b,
                    color_discrete_sequence=['rgba(150, 150, 150, 0.5)']
                    ).data[0],
                )


            st.plotly_chart(fig2)

    else:
        st.write('Routine file not available')

st.sidebar.markdown('## Rank Analysis')
if (event_str == 'All') | (gender == 'All'):
    st.sidebar.markdown('*Select event and gender for rank analysis*')

phase = st.sidebar.selectbox(
     'Select Phase:',
     (df_select["Phase"].unique().tolist() )
    )

sql_str = "SELECT * from " + "(" + sql_str + ") AS T " + "where Phase=" + "'" + phase + "'"


df_ranking = make_call(sql_str, engine)

rating = st.sidebar.selectbox(
     'Select Rating:',
     (rating_str )
    )
if (event_str != 'All') & (gender != 'All'):
    st.sidebar.markdown('*Select different ratings to see rank changes*')

if (event_str != 'All') and (gender != 'All') and (athlete == 'All'):
    # st.write(rating)
    df_result = calculate_ranks(df_ranking, rating)
    st.markdown("### Rank Analysis")
    st.table(df_result)

