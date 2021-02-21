#python code for parallel computing
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import random
import joblib
import pickle, joblib
import autoloader

pickle_prefix = "LottoMAX_df"
pickle_df_path="pickle/df"

def GetFileID(filepath):
    file_id = filepath[filepath.rfind("_"):filepath.rfind(".")]  # e.g., "_6"
    return file_id

def GetVariableName(obj, namespace=globals()): #get name of the variable as a string
    return [name for name in namespace if namespace[name] is obj][0]

def SavePickle(relative_path="", pickle=None, rename=None):
    var_name = GetVariableName(pickle)
    if rename != None:
        var_name = rename
    pickle_name = var_name + ".pkl"
    autoloader.create_folder(relative_path)
    joblib.dump(pickle, os.path.join(os.getcwd(), relative_path, pickle_name))
    print("<" + GetVariableName(pickle) + ">", "saved as pickle:", os.path.join(relative_path, pickle_name))


def LoadPickle(relative_path="", pickle_name=None, verbose=1):
    try:
        x = joblib.load(os.path.join(os.getcwd(), relative_path, pickle_name))
        if verbose > 0:
            print("Pickle loaded:", os.path.join(relative_path, pickle_name))
        return x
    except:
        print("Cannot find pickle:", os.path.join(relative_path, pickle_name))
        return None

def RunLotto(min=1, max=49, numbers=7, sets=3, win_limit=10, save_interval=10,
             pickle_path=pickle_df_path, pickle_name=pickle_prefix):
    print("▁▂▃▄▅▆▇ Lotto MAX (IQ Tax) Ultimate Simulator (TM) initialized! █▇▆▅▄▃▂▁" +
          "\nRange of numbers: [" + str(min) + ", " + str(max)+"]"+
          "\nLength of sequence: "+str(numbers) +
          "\nNumber of sequences in a ticket: "+str(sets)+
          "\nSimulation endpoint: "+str(win_limit)+" wins"+
          "\nAutosave interval: "+str(save_interval)+"s"+
          "\nSave path: "+str(os.path.join(pickle_path, pickle_name)))
    t0 = datetime.now()
    t_last_save = t0
    df = pd.DataFrame(columns=["won","numbers","tickets_count"])
    df.index.name="gambler ID"
    win_count = 0
    current_row = str(datetime.now()) #use time string as ID to make each row unique
    performance_measure = 0
    pickle_name += ".pkl"
    pickle = os.path.join(pickle_path, pickle_name)
    autoloader.create_folder(pickle_path)
    try:
        df = pd.read_pickle(pickle)
        print("Pickle loaded as df:", pickle)
        #current_row = df[df["won"]==0].index[0] #go to the unfinished simulation from last time
        # this might introduce some bias (if the task has been interrupted many times), but better than discarding the computational resource
    except:
        print("No existing pickle found, will create new pickle for the df")
    print("✔ Simulation started!!!!")
    t1 = datetime.now()
    save_now = False
    while win_count < win_limit:    #loop for the current session until certain number of data points are acquired
        tickets_bought = 0
        try:
            current_row = df[df["won"] == 0].index[0]  # finish the unfinished simulation if there is any
            tickets_bought = df.at[current_row, "tickets_count"]  # continue counting for unfinished simulations
        except:
            current_row = str(datetime.now()) #new row
            tickets_bought = 0
        current_win = 0
        while current_win == 0: #loop for a single buyer until he wins
            winning_numbers = random.sample(range(min, max+1), numbers)
            winning_numbers.sort()
            for i in range(0, sets): #1 Lotto Max ticket has 3 sets of numbers by default, so it's counted as 1 step
                x = random.sample(range(min, max+1), numbers)
                x.sort()
                if winning_numbers == x:
                    current_win += 1 #there could be multiple wins in one ticket
                    save_now = True
                    win_count += 1
                    print("★\tCongratulations! You have won the Lotto MAX!!! Winning numbers:", winning_numbers)
            tickets_bought += 1
            performance_measure += 1
            if (datetime.now()-t1).total_seconds() > save_interval:
                save_now = True
                t1=datetime.now()
            if save_now == True:
                df.at[current_row, "tickets_count"] = tickets_bought
                df.at[current_row, "won"] = current_win
                df["numbers"] = df["numbers"].astype(object)  # stores the winning sequence as a list in the cell
                df.at[current_row, "numbers"] = winning_numbers
                print("▶▶▶ Auto-saving...\tTime elapsed: " + str(datetime.now() - t0))
                df.to_pickle(pickle)
                print("Current performance: %.1f tickets/sec"%(performance_measure/(datetime.now()-t_last_save).total_seconds())+
                      "\tProcessing ID:",repr(current_row)+"\tTickets bought:", tickets_bought)
                print("Current winning count: " + str(win_count) + " / " +str(win_limit))
                t_last_save = datetime.now()
                performance_measure = 0
                save_now = False
    print("(◕ᴗ◕✿)\tAll simulations complete ("+str(win_limit)+" wins reached)! Current table size: "+str(df.shape)+
          "\nTotal time elapsed: " + str(datetime.now() - t0))

