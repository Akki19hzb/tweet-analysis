
class config:
    # name of the data pkl created
    data_pkl_name = None
    # name of the model pickle
    name_of_model = None
    # name of the excel file
    name_of_excel = None
    # name of label encoder
    label_encoder_name = None
    # current strike rate
    current_sr = None
    # average
    avg = None

    # average number of balls played
    no_of_balls_avg = None

    # total number of match

    total_number_of_match = None

    total_runs = None
    total_50 = None
    total_100 = None






class Sachin_config(config):

     # name of the data pkl created
     data_pkl_name='data.pkl'
     # name of the model pickle
     name_of_model='sachin-model.pkl'
     # name of the excel file
     name_of_excel='Sachin-ODI.xlsx'
     #name of label encoder
     label_encoder_name='sachin_lable.pkl'
     # current strike rate
     current_sr=31.5
     # average
     avg=81.4

     #average number of balls played
     no_of_balls_avg=47

     # total number of match

     total_number_of_match=463

     total_runs=18426
     total_50=96
     total_100=49




class Kholi_config(config):
    # TODO Make changes in the file.
    name_of_model = 'kholi-model.pkl'
    name_of_excel = 'kholi-ODI.xlsx'
    total_runs = 18000
    total_50 = 34
    total_100 = 50