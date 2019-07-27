
# coding: utf-8

# In[31]:


import pandas as pd
data=pd.read_csv('nba.csv',index_col=False)


# In[32]:


data=data.drop(['Unnamed: 6','Unnamed: 7','Notes'],axis=1)
data=data.rename(columns={'PTS.1':'PTS_1'})

# 30 features , each feature represents a team, if engage, 1. else 0.
# 1 feature, the previous team is playing at home or away. home 1 else 0.


# In[33]:


def who_win(a,b):
    if a > b:
        return 1
    else:
        return 0
data['visitor_win']=data.apply(lambda x: who_win(x.PTS, x.PTS_1), axis = 1)


# In[34]:


data.head(10)


# In[35]:


team_index={'Atlanta Hawks':0,'Boston Celtics':1,'Brooklyn Nets':2,'Charlotte Hornets':3,'Charlotte Bobcats':3,
           'Chicago Bulls':4,'Cleveland Cavaliers':5,'Dallas Mavericks':6,'Denver Nuggets':7,'Detroit Pistons':8,
           'Golden State Warriors':9,'Houston Rockets':10,'Indiana Pacers':11,'Los Angeles Clippers':12,
           'Los Angeles Lakers':13,'Memphis Grizzlies':14,'Miami Heat':15,'Milwaukee Bucks':16,'Minnesota Timberwolves':17,
           'New Orleans Pelicans':18,'New York Knicks':19,'Oklahoma City Thunder':20,'Orlando Magic':21,'Philadelphia 76ers':22,
           'Phoenix Suns':23,'Portland Trail Blazers':24,'Sacramento Kings':25,'San Antonio Spurs':26,'Toronto Raptors':27,
           'Utah Jazz':28,'Washington Wizards':29}
data_for_ba seline=data[['Start (ET)','Visitor','PTS','Home','PTS_1']]
data_for_baseline.head(10)


# In[36]:


import numpy as np
y=np.array(data['visitor_win'])
y #label -- 客队是否赢球


# In[37]:


#30个 0，1 表示参赛球队，1个feature表示在前面的那队是不是客队
X=[]

for index, row in data_for_baseline.iterrows():
    x=np.zeros(31)
    i_visitor=team_index[row['Visitor']]
    i_home=team_index[row['Home']]
    x[i_visitor]=1
    x[i_home]=1
    if i_visitor < i_home:
        x[30]=1
    X.append(x)


# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[39]:


from sklearn.svm import SVC

svm=SVC(random_state=42,probability=True,gamma='auto')
svm.fit(X_train,y_train)
svm.score(X_test,y_test)


# In[40]:


# Load Kaggle data
all_player_df = pd.read_csv("Seasons_Stats.csv")
all_player_df = all_player_df.loc[all_player_df['Year'].isin([2017,2016,2015,2014,2013])].reset_index()
all_player_df = all_player_df.drop(['index','Unnamed: 0','Year'], axis=1).T
all_player_df = all_player_df.rename(columns=all_player_df.iloc[0]).drop(all_player_df.index[0]) 
# pd.set_option('display.max_columns', 500)
# print (all_player_df.ix[:, (all_player_df == 'PHI').any()])
all_player_df.head(10)


# In[42]:


career=pd.read_csv('player_career.csv',index_col=False)
pd.set_option('display.max_columns', None)
career = career.dropna(subset=['3P','GS'])
career = career.T
career = career.rename(columns=career.iloc[0]).drop(career.index[0]).dropna(how='all').drop('Lg')

career = career.rename(columns={'Nene': 'Nene Hilario', 'Jeff Taylor':'Jeffery Taylor'})
career=career.fillna(0)

# print (career.ix[:, (career == 926).any()])


# In[13]:


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
            #print(name,temp)
            select.append(temp[5])
            select.append(temp[12])
            select.append(temp[15])
            select.extend(temp[18:])
            feature.extend(select) 
        feature = np.array(feature)
        x_train.append(feature)

    
    return (np.array(x_train), np.array(y_train))                       


# In[14]:


X, y = feature_prep(data, career)


# In[15]:


X.shape


# In[16]:


for i in range(len(X)):
    array = np.array(X[i], dtype=np.float64, order='C')


# In[17]:


data.iloc[4966]


# In[18]:


X[4966]


# In[19]:


career['Miles Plumlee']


# In[20]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print (X_train.shape)
print (y_train.shape)
svm=SVC(random_state=42,probability=True,gamma='auto')
svm.fit(X_train,y_train)
svm.score(X_test,y_test)


# In[25]:


import numpy as np
def feature_prep_all(game_df, player_df):
    
    y_train = np.array(game_df['visitor_win'])
    
    stat_names = np.array(player_df.index.values)
    
    x_train = []
    for index, row in game_df.iterrows():
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
            select.extend(temp)
            #for idx in features_idx_lst: 
                #select.append(temp[idx])
            feature.extend(select) 
        feature = np.array(feature)
        x_train.append(feature)
   
    return (np.array(x_train), np.array(y_train))                       
X_all,y_all = feature_prep_all(data, career)


# In[26]:



# Find the number 5th statistics of all ten playes out of 25 statistics
X_all[0][4::25]


# In[27]:


feature_names = [ 'G', 'GS', 'MP', 'FG',
                 'FGA','FG%','3P','3PA', '3P%',
                 '2P','2PA','2P%','eFG%','FT', 
                 'FTA' ,'FT%','ORB','DRB','TRB',
                 'AST', 'STL', 'BLK', 'TOV','PF', 'PTS']

feature_names_idx_dict = { 'G':0, 'GS':1, 'MP':2, 'FG':3,
                 'FGA':4,'FG%':5,'3P':6,'3PA':7, '3P%':8,
                 '2P':9,'2PA':10,'2P%':11,'eFG%':12,'FT':13, 
                 'FTA':14 ,'FT%':15,'ORB':16,'DRB':17,'TRB':18,
                 'AST':19, 'STL':20, 'BLK':21, 'TOV':22,'PF':23, 'PTS':24}
all_feature_names = []
for idx in range(10):
    for feature_name in feature_names:
        feature_name_idx = feature_name+"_"+str(idx)
        all_feature_names.append(feature_name_idx)
all_feature_names


# In[28]:


def forword_select(data, career, select_num):
    feas_select = []
    X = []
    scores_all = []
    
    for i in range(5):
        feas_left  = list(set(feature_names) - set(feas_select))
        max_score = 0
        print('feas_select = ',feas_select )
        scores = {}
        for fea in feas_left:
            print('test fea = ',fea)
            new_idx = feature_names_idx_dict[fea]
            #print('new idx = ',new_idx)
            tmp_lst = []
            for item in X_all:
                tmp_lst.append(item[new_idx::25])
            X_tmp = np.array(tmp_lst)
            print('len x tmp[0] = ',len(X_tmp[0]))
            X_train, X_test, y_train, y_test = train_test_split(X_tmp,y_all,test_size=0.3,random_state=42)
            svm.fit(X_train,y_train)
            score = svm.score(X_test,y_test)
            scores[fea] = score
            print('score = ',score)
            if (score > max_score):
                max_score = score
                best_fea = fea
        feas_select.append(best_fea)
        
        best_idx = feature_names_idx_dict[best_fea]
        best_lst = []
        for item in X_all:
            best_lst.append(item[best_idx::25])
        X_best = np.array(best_lst)
        if len(X) == 0:
            X = X_best
        else:
            X = np.concatenate((X, X_best),axis = 1)
        print('i = ',i)
        print('best_fea = ',best_fea)
        print('max_score = ',max_score)
        print('scores = ',scores)
        scores_all.append(scores)
        #feas_select.append(best_fea)
    print('feas_select = ', feas_select)
    return feas_select  


# In[29]:


forword_select(data, career,4)


# In[43]:


def backword_select(data, career):
    feas_removed = []
    X = []
    scores_all = []
    
    for i in range(5):
        feas_left  = list(set(feature_names) - set(feas_removed))
        min_score = 1
        print('feas_removed = ',feas_removed )
        scores = {}
        X = X_all
        for fea in feas_left:
            print('current fea to be removed = ',fea)
            new_idx = feature_names_idx_dict[fea]
            try_removed_idxs = []
            for i in range(1,26):
                try_removed_idxs.append(new_idx * i)
            X_tmp = np.delete(X,try_removed_idxs,1)
            print('len x[0] = ',len(X[0]))
            print('len x tmp[0] = ',len(X_tmp[0]))
            X_train, X_test, y_train, y_test = train_test_split(X_tmp,y_all,test_size=0.3,random_state=42)
            svm.fit(X_train,y_train)
            score = svm.score(X_test,y_test)
            scores[fea] = score
            print('score = ',score)
            if (score < min_score):
                min_score = score
                worst_fea = fea
        feas_removed.append(worst_fea)
        
        worst_idx = feature_names_idx_dict[worst_fea]
        removed_idxs = []
        for i in range(1,26):
            removed_idxs.append(worst_idx * i)
        X = np.delete(X,removed_idxs,1)
        print('i = ',i)
        print('worst_fea = ',worst_fea)
        print('min_score = ',min_score)
        print('scores = ',scores)
        scores_all.append(scores)
        #feas_select.append(worst_fea)
    print('feas_removed = ', feas_removed)
    print('feas_left = ', feas_left)
    return feas_left  
backword_select(data, career)

