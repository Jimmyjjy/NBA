
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data=pd.read_csv('nba.csv',index_col=False)


# In[3]:


data=data.drop(['Unnamed: 6','Unnamed: 7','Notes'],axis=1)
data=data.rename(columns={'PTS.1':'PTS_1'})


# 30 features , each feature represents a team, if engage, 1. else 0.
# 1 feature, the previous team is playing at home or away. home 1 else 0.
# playing time. how to represent? 


# In[4]:


def who_win(a,b):
    if a > b:
        return 1
    else:
        return 0
data['visitor_win']=data.apply(lambda x: who_win(x.PTS, x.PTS_1), axis = 1)


# In[5]:


data.head(10)


# In[6]:


team_index={'Atlanta Hawks':0,'Boston Celtics':1,'Brooklyn Nets':2,'Charlotte Hornets':3,'Charlotte Bobcats':3,
           'Chicago Bulls':4,'Cleveland Cavaliers':5,'Dallas Mavericks':6,'Denver Nuggets':7,'Detroit Pistons':8,
           'Golden State Warriors':9,'Houston Rockets':10,'Indiana Pacers':11,'Los Angeles Clippers':12,
           'Los Angeles Lakers':13,'Memphis Grizzlies':14,'Miami Heat':15,'Milwaukee Bucks':16,'Minnesota Timberwolves':17,
           'New Orleans Pelicans':18,'New York Knicks':19,'Oklahoma City Thunder':20,'Orlando Magic':21,'Philadelphia 76ers':22,
           'Phoenix Suns':23,'Portland Trail Blazers':24,'Sacramento Kings':25,'San Antonio Spurs':26,'Toronto Raptors':27,
           'Utah Jazz':28,'Washington Wizards':29}
data_for_baseline=data[['Start (ET)','Visitor','PTS','Home','PTS_1']]
data_for_baseline.head(10)


# In[7]:


import numpy as np
y=np.array(data['visitor_win'])
y #label -- Whether away team win or not


# In[ ]:


#30 Boolean 0，1 

for index, row in data_for_baseline.iterrows():
    x=np.zeros(31)
    i_visitor=team_index[row['Visitor']]
    i_home=team_index[row['Home']]
    x[i_visitor]=1
    x[i_home]=1
    if i_visitor < i_home:
        x[30]=1
    X.append(x)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[ ]:


from sklearn.svm import SVC

svm=SVC(random_state=42,probability=True,gamma='auto')
svm.fit(X_train,y_train)
svm.score(X_test,y_test)


# In[5]:


career=pd.read_csv('player_career.csv',index_col=False)
pd.set_option('display.max_columns', None)
career = career.dropna(subset=['3P','GS'])
career = career.T
career = career.rename(columns=career.iloc[0]).drop(career.index[0]).dropna(how='all').drop('Lg')

career = career.rename(columns={'Nene': 'Nene Hilario', 'Jeff Taylor':'Jeffery Taylor'})
career=career.fillna(0)

career['Michael Jordan']
# print (career.ix[:, (career == 926).any()])


# In[ ]:


career


# In[6]:


import numpy as np
def feature_prep(game_df, player_df):
    
    y_train = np.array(game_df['visitor_win'])
    
    stat_names = np.array(player_df.index.values)
    
    x_train = []
    for index, row in game_df.iterrows():
#         feature_dict = {}
        feature = []
        names = row[7:17]
        for name in names:
            select = []
            if name == 'Luigi Datome': name = 'Gigi Datome'
            if name == 'Kelly Oubre': name = 'Kelly Oubre Jr.'
            if name == 'Derrick Jones': name = 'Derrick Jones Jr.'
            if name == 'Taurean Waller-Prince': name = 'Taurean Prince'
            if name == 'Dennis Smith': name = 'Dennis Smith Jr.'
            if name == 'Frank Mason': name = 'Frank Mason III'
            player_stat = player_df[name].values
            temp = [stat for stat in player_stat]
            select.append(temp[5])
            select.append(temp[12])
            select.extend(temp[15:])
            feature.extend(select)
        x_train.append(feature)
#             feature_dict[name] = {}
#             player_stat = player_df[name].values
#             for i in range(len(player_stat)):
#                 feature_dict[name][stat_names[i]] = player_stat[i]
#         x_train.append(feature_dict)
    
    #print(x_train[0])
    
    return (np.array(x_train), np.array(y_train))                       


# In[7]:


X, y = feature_prep(data, career)


# In[8]:


from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)
dt6 = DecisionTreeClassifier(max_depth=6, random_state=42)
dt6.fit(X_train, y_train)
dt6.score(X_test,y_test)


# In[9]:


from sklearn.ensemble import BaggingClassifier

bag1=BaggingClassifier(n_estimators=31,random_state=314,base_estimator=dt6)
bag1.fit(X_train,y_train)
bag1.score(X_test,y_test)


# In[ ]:


for i in range(len(X)):
    array = np.array(X[i], dtype=np.float64, order='C')


# In[10]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print (X_train.shape)
print (y_train.shape)
svm=SVC(random_state=42)
svm.fit(X_train,y_train)
svm.score(X_test,y_test)


# In[11]:


#What if Kobe Bryant still plays for Lakers?

#2016-2017 season
LA_2016=data[(data.Home=='Los Angeles Lakers') | (data.Visitor=='Los Angeles Lakers')][246:328]

def switch(x):
    for i in range(len(x)):
        if x[i]=='Nick Young' or x[i]=='Lou Williams' or x[i]=='David Nwaba' or x[i]=='Jordan Clarkson': # 4 SG in Lakers roster that year
            x[i]='Kobe Bryant'
            break
    return x
LA_2016[['v_player1','v_player2','v_player3','v_player4','v_player5',
       'h_player1','h_player2','h_player3','h_player4','h_player5']]=LA_2016.apply(lambda x: switch(x[7:17]),axis=1)

x_predict,y_useless=feature_prep(LA_2016,career)
y_predict=bag1.predict(x_predict)

visitor=list(LA_2016[['Visitor']].values)
win=0
for i in range(len(visitor)):
    if visitor[i][0]=='Los Angeles Lakers' and y_predict[i]==1:
        win+=1
    elif visitor[i][0]!='Los Angeles Lakers' and y_predict[i]==0:
        win+=1
        
def count_app(x):
    count=0
    if x=='Kobe Bryant':
        count+=1
    return count
app=np.sum(np.array(LA_2016.applymap(count_app)))        
        
print('Lakers could get %d wins, %d losts with Kobe Bryant starting %d games in 2016-2017 season'%(win,82-win,app))
print('While they actually got 26 wins, 56 losts in that season')


# In[ ]:


LA_2016.head(10)


# In[12]:


#2017-2018 season
LA_2017=data[(data.Home=='Los Angeles Lakers') | (data.Visitor=='Los Angeles Lakers')][328:410]

def switch(x):
    for i in range(len(x)):
        if x[i]=='Vander Blue' or x[i]=='Kentavious Caldwell-Pope' or x[i]=='Josh Hart' or x[i]=='Jordan Clarkson' or x[i]=='Andre Ingram': # 4 SG in Lakers roster that year
            x[i]='Kobe Bryant'
            break
    return x
LA_2017[['v_player1','v_player2','v_player3','v_player4','v_player5',
       'h_player1','h_player2','h_player3','h_player4','h_player5']]=LA_2017.apply(lambda x: switch(x[7:17]),axis=1)

x_predict,y_useless=feature_prep(LA_2017,career)
y_predict=bag1.predict(x_predict)

visitor=list(LA_2017[['Visitor']].values)
win=0
for i in range(len(visitor)):
    if visitor[i][0]=='Los Angeles Lakers' and y_predict[i]==1:
        win+=1
    elif visitor[i][0]!='Los Angeles Lakers' and y_predict[i]==0:
        win+=1
        
def count_app(x):
    count=0
    if x=='Kobe Bryant':
        count+=1
    return count
app=np.sum(np.array(LA_2017.applymap(count_app)))        
        
print('Lakers could get %d wins, %d losts with Kobe Bryant starting %d games in 2017-2018 season'%(win,82-win,app))
print('While they actually got 35 wins, 47 losts in that season')


# In[ ]:


#What if Michael Jordan's Bulls plays in today's league?
#What if Draymond Green hasn't got injured in 2015-2016's finals?
#What if T-mac and Yao's Rockets plays in today's league?
#Is Carmelo Anthony really unsuitable for Rockets?
#What if a team owns 5 Lebron James?
#Can Golden State Warriors defeat Cavaliers these years without Kevin Durant?
#What will be the final ranking of this regular season?


# In[13]:


#What if Michael Jordan's Bulls plays in today's league?

chi_2017=data[(data.Home=='Chicago Bulls') | (data.Visitor=='Chicago Bulls')][246:328]

def back_to_1996(x):
    if x[2] == 'Chicago Bulls': 
        x[7],x[8],x[9],x[10],x[11]='Michael Jordan','Scottie Pippen','Dennis Rodman','Luc Longley','Ron Harper'
    if x[4] == 'Chicago Bulls': 
        x[12],x[13],x[14],x[15],x[16]='Michael Jordan','Scottie Pippen','Dennis Rodman','Luc Longley','Ron Harper'
    return x
chi_2017[['Date','Start (ET)','Visitor','PTS','Home','PTS_1','Attend.','v_player1','v_player2','v_player3','v_player4','v_player5',
       'h_player1','h_player2','h_player3','h_player4','h_player5']]=chi_2017.apply(lambda x: back_to_1996(x[0:17]),axis=1)


x_predict,y_useless=feature_prep(chi_2017,career)
y_predict=svm.predict(x_predict)

visitor=list(chi_2017[['Visitor']].values)
win=0
for i in range(len(visitor)):
    if visitor[i][0]=='Chicago Bulls' and y_predict[i]==1:
        win+=1
    elif visitor[i][0]!='Chicago Bulls' and y_predict[i]==0:
        win+=1

print('Chicago Bulls could get %d wins, %d losts in 2016-2017 season with their starting fives back in 1995-1996 season'%(win,82-win))
print('While they actually got 72 wins, 10 losts back in 1995-1996')



# In[ ]:


chi_2017.head(10)


# In[14]:


#What will be the final ranking of this regular season?

starters_2018=pd.read_csv('2018_starters.csv',header=None)
starters_2018=starters_2018.drop(columns=6)

starters_2018_dict={}
for index, row in starters_2018.iterrows():
    starters_2018_dict[row[0]]=list(row[1:].values)


# In[15]:


data_2019=pd.read_csv('2019_schedule.csv',index_col=False)
data_2019=data_2019.drop(['Unnamed: 6','Unnamed: 7','Notes'],axis=1)
data_2019=data_2019.rename(columns={'PTS.1':'PTS_1'})
data_2019['visitor_win']=data_2019.apply(lambda x: who_win(x.PTS, x.PTS_1), axis = 1)


# In[16]:


data_2019_fut=data_2019[386:] #unplayed matches


# In[17]:


def start(x):
    v=x[2]
    #print(v)
    h=x[4]
    #print(h)
    for i in range(7,12):
        #print(starters_2018_dict[v][i-7])
        x[i]=starters_2018_dict[v][i-7]
    for i in range(12,17):
        x[i]=starters_2018_dict[h][i-12]
    
    return x

data_2019_fut[['Date','Start (ET)','Visitor','PTS','Home','PTS_1','Attend.','v_player1','v_player2','v_player3','v_player4','v_player5',
       'h_player1','h_player2','h_player3','h_player4','h_player5']]=data_2019_fut.apply(lambda x: start(x[0:17]),axis=1)


# In[18]:


x_predict,y_useless=feature_prep(data_2019_fut,career)
y_predict=bag1.predict(x_predict)

win_record={i:0 for i in set(data_2019_fut['Visitor'].values)}

for i in range(len(y_predict)):
    v=data_2019_fut.loc[386+i].Visitor
    h=data_2019_fut.loc[386+i].Home
    if y_predict[i]==1:
        win_record[v]+=1
    else:
        win_record[h]+=1


# In[19]:


#real record by 12/08/2018
east_now={'Toronto Raptors':[21,6],'Philadelphia 76ers':[18,9],'Milwaukee Bucks':[16,8],
         'Indiana Pacers':[16,10],'Boston Celtics':[15,10],'Detroit Pistons':[13,10],'Charlotte Hornets':[12,13],
         'Orlando Magic':[12,14],'Miami Heat':[11,14],'Washington Wizards':[11,15],'Brooklyn Nets':[10,18],
         'New York Knicks':[8,19],'Atlanta Hawks':[6,20],'Cleveland Cavaliers':[6,20],'Chicago Bulls':[6,21]}
west_now={'Golden State Warriors':[18,9],'Oklahoma City Thunder':[16,8],'Denver Nuggets':[17,9],
         'Los Angeles Clippers':[16,9],'Los Angeles Lakers':[16,10],'Memphis Grizzlies':[15,10],
         'Portland Trail Blazers':[15,11],'Dallas Mavericks':[13,11],'Sacramento Kings':[13,12],
         'Utah Jazz':[13,13],'Minnesota Timberwolves':[13,13],'New Orleans Pelicans':[13,14],
             'San Antonio Spurs':[12,14],'Houston Rockets':[11,14],'Phoenix Suns':[4,22]}
#predicted record
east={}
west={}

for key,val in east_now.items():
    east[key]=val[0]+win_record[key]
for key,val in west_now.items():
    west[key]=val[0]+win_record[key]


# In[47]:


east_now_df=pd.DataFrame(east_now,index=['Win','Lost']).T
east_now_df['W%']=east_now_df.apply(lambda x: x.Win/(x.Win+x.Lost), axis=1)
east_now_df=east_now_df.sort_values(by=['W%'],ascending=False)

east_now_df_1=east_now_df.reset_index()
east_now_df_1=east_now_df_1.rename(columns={'index':'name'})


new_index=[]
for key in east_now_df.index.values[:8]:
    new_index.append(key+"*")
new_index.extend( east_now_df.index.values[8:])
east_now_df.index=new_index

print("By 12/08/2018, the real eastern conference standing (*playoffs):")
display(east_now_df)

east_df=pd.DataFrame(east,index=['Win']).T
east_df['Lost']=east_df.apply(lambda x: 82-x.Win, axis=1)
east_df['W%']=east_df.apply(lambda x: x.Win/82, axis=1)
east_df=east_df.sort_values(by=['Win'],ascending=False)


new_index=[]
east_df_1=east_df.reset_index()
east_df_1=east_df_1.rename(columns={'index':'name'})

count=0
for name in east_df_1.name.values:
    pre_rank=east_now_df_1[east_now_df_1.name==name].index.values[0]
    if count<8:
        if count==pre_rank:
            new_index.append(name+'*  (=)' )
        elif  count>pre_rank:
            new_index.append(name+'*  ('+str(pre_rank-count)+')')
        else:
            new_index.append(name+'*  (+'+str(pre_rank-count)+')')
    else:
        if count==pre_rank:
            new_index.append(name+'  (=)' )
        elif  count>pre_rank:
            new_index.append(name+'  ('+str(pre_rank-count)+')')
        else:
            new_index.append(name+'  (+'+str(pre_rank-count)+')')
    count+=1
east_df.index=new_index

print("Our model predicts that the final eastern conference standing in 2018-2019 season (*playoffs):")
display(east_df)


# In[48]:


west_now_df=pd.DataFrame(west_now,index=['Win','Lost']).T
west_now_df['W%']=west_now_df.apply(lambda x: x.Win/(x.Win+x.Lost), axis=1)
west_now_df=west_now_df.sort_values(by=['W%'],ascending=False)

west_now_df_1=west_now_df.reset_index()
west_now_df_1=west_now_df_1.rename(columns={'index':'name'})


new_index=[]
for key in west_now_df.index.values[:8]:
    new_index.append(key+"*")
new_index.extend( west_now_df.index.values[8:])
west_now_df.index=new_index

print("By 12/08/2018, the real western conference standing (*playoffs):")
display(west_now_df)

west_df=pd.DataFrame(west,index=['Win']).T
west_df['Lost']=west_df.apply(lambda x: 82-x.Win, axis=1)
west_df['W%']=west_df.apply(lambda x: x.Win/82, axis=1)
west_df=west_df.sort_values(by=['Win'],ascending=False)


new_index=[]
west_df_1=west_df.reset_index()
west_df_1=west_df_1.rename(columns={'index':'name'})

count=0
for name in west_df_1.name.values:
    pre_rank=west_now_df_1[west_now_df_1.name==name].index.values[0]
    if count<8:
        if count==pre_rank:
            new_index.append(name+'*  (=)' )
        elif  count>pre_rank:
            new_index.append(name+'*  ('+str(pre_rank-count)+')')
        else:
            new_index.append(name+'*  (+'+str(pre_rank-count)+')')
    else:
        if count==pre_rank:
            new_index.append(name+'  (=)' )
        elif  count>pre_rank:
            new_index.append(name+'  ('+str(pre_rank-count)+')')
        else:
            new_index.append(name+'  (+'+str(pre_rank-count)+')')
    count+=1
west_df.index=new_index

print("Our model predicts that the final western conference standing in 2018-2019 season (*playoffs):")
display(west_df)

