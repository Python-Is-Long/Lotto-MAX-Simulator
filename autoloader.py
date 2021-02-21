import os
import pandas as pd
def create_folder(folder):
    if os.path.exists(folder)==0 and folder!="":
        try:
            os.makedirs(folder)  # auto create folder if it doesn't exist
        except:
            print("\n(꒪Д꒪)ノ\tPATH ERROR -- cannot create folder:", folder)

def load_csv(file_path, header='infer', column_names=None, pickle_path=""):
    df = pd.DataFrame()
    load_mode = -1
    file = os.path.basename(file_path) #"Features_Variant_1.csv"
    file_name = os.path.splitext(file)[0] #"Features_Variant_1"
    pickle_name = file_name+".pkl"
    pickle = os.path.join(pickle_path, pickle_name)
    try:
        df = pd.read_pickle(pickle)
        load_mode = 1
    except:
        df = pd.read_csv(file_path, header=header, names=column_names)
        create_folder(pickle_path)
        df.to_pickle(pickle)
        load_mode = 0
    if load_mode==1:
        print("Existing pickle found and loaded: "+pickle)
    elif load_mode==0:
        print("CSV loaded: "+file_path+", saved as pickle: "+pickle)
    return df