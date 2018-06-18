import pandas,pickle,os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from cricket import load_from_pickle,save_into_pickle
from config import Sachin_config,Kholi_config,config
from random import randint


def simulate_n_matches(model,config, n_matches):
    # we loop over the total matches and calculate the total run.
    total_run=config.total_runs
    avg=config.avg
    sr=config.current_sr
    total_matches=config.total_number_of_match
    total_50=config.total_50
    total_100=config.total_100
    for i in range(0,n_matches):
        # we first randomly select a opponent
        le=load_from_pickle(config.label_encoder_name)
        rand_opponent=randint(0,len(le.classes_)-1)
        # we convert the shape to a row so that model dont give any warning
        temp=np.array([rand_opponent,avg,sr]).reshape(1, -1)

        run=model.predict(temp)
        total_run+=run
        total_matches+=1

        if run>=50 and run <100:
            total_50+=1
        elif run>=100:
            total_100+=1


        # now update the sr
        sr=(total_run/(config.no_of_balls_avg*total_matches))*100

        # now we update the avg
        avg=total_run/total_matches

        print('\n')
        print('---' * 5)
        print('Match No=%s' % config.total_number_of_match)
        print('Opponent=%s.' % (le.inverse_transform(rand_opponent)))
        print('Avg=%s.' % (avg))
        print('Strike rate=%s.' % sr)
        print('Run=%s' % run)
        print('---' * 5)

    # In the end we update the new values .
        config.avg=avg
        config.current_sr=sr
        config.total_runs=total_run
        config.total_number_of_match=total_matches
        config.total_100=total_100
        config.total_50=total_50

def print_current_stats(config_obj):
     print('*'*10)
     print('Total Runs=%s'%config_obj.total_runs)
     print('Current SR=%s'%config_obj.current_sr)
     print('Current Avg=%s'%config_obj.avg)
     print('Total 50=%s'%config_obj.total_50)
     print('Total 100=%s' % config_obj.total_100)
     print('*'*10)








player_selection=input('Please select the player you want to analyse.\n1.Sachin\n2.Kholi\n')
model=None

if player_selection=='1':
    model=load_from_pickle(Sachin_config.name_of_model)
elif player_selection=='2':
    #TODO make the kholi-model.pkl from cricket.py
    model=load_from_pickle('kohli-model.pkl')
else:
    print('Please enter the req input.')
    exit()

selection=100

while selection!='3':
    selection=input('Select the type of analysis.\n1.Simulate N matches'
                '\n2.Print current stats.'
                '\n3.Exit')
    if selection=='1':
        n_matches=input('Enter the number of matches to simulate.\n')
        if player_selection=='1':
         simulate_n_matches(model,Sachin_config,int(n_matches))
        if player_selection=='2':
          simulate_n_matches(model,Kholi_config,int(n_matches))
    elif selection=='2':
        if player_selection=='1':
         print_current_stats(Sachin_config)
        if player_selection=='2':
          print_current_stats(Kholi_config)
    elif selection=='3':
        print('Exit')
        exit()
