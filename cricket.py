import pandas,pickle,os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from config import Sachin_config
import matplotlib.pyplot as plt

"""
 This file is used for model training.
"""


data_pkl_name=Sachin_config.data_pkl_name
model_pkl_name=Sachin_config.name_of_model

excel_file_name=Sachin_config.name_of_excel

lable_encoder_name=Sachin_config.label_encoder_name


def show_plot(x,y,z,label1='X-AXIS',label2='Y-AXIS',label3='Z-AXIS'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel(label3)

    plt.show()

def save_into_pickle(python_object,pickle_name):

    pkl_file=open(pickle_name,'wb')
    pickle.dump(python_object,pkl_file)
    pkl_file.close()

def load_from_pickle(pickle_name):
    if os.path.exists(pickle_name):
        pkl_file=open(pickle_name,'rb')
        obj=pickle.load(pkl_file)
        pkl_file.close()
        return obj
    else:
        raise Exception('No pickle file found.')


if __name__ == "__main__":

    if os.path.exists(data_pkl_name):
        df = load_from_pickle(data_pkl_name)
    else:
        df = pandas.read_excel(excel_file_name)
        # we check if we have the pickle object,if not we read the file again.
        save_into_pickle(df,data_pkl_name)

    # removes all the Null values
    df=df.dropna()

    # input parameters
    x=df[['Versus','Avg','S/R']]
    # output parameters
    y=df[['Runs']]

    # label encoder is used to convert the string names to appropriate numbers
    le=LabelEncoder()
    x['Versus']=le.fit_transform(x['Versus'])

    save_into_pickle(le, lable_encoder_name)

    X_train=x
    y_train=y

    # TODO READ ABOUT THE ALGO
    model=RandomForestRegressor(n_estimators=10,random_state=0)
    model.fit(X_train,y_train)

    y_pred=model.predict(X_train)

    # todo r2 score
    print('Accuracy=%s%%'%(r2_score(y_train, y_pred)*100))

    # we now save the model into a pickle file.

    save_into_pickle(model,model_pkl_name)









