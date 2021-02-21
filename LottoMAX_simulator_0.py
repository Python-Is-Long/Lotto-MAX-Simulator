#copy oc code for parallel computing
import LottoMAX_engine

pickle_name = LottoMAX_engine.pickle_prefix+LottoMAX_engine.GetFileID(__file__)

def main():
    LottoMAX_engine.RunLotto(min=1, max=49, numbers=7, sets=3, win_limit=1000, save_interval=10,
                             pickle_path=LottoMAX_engine.pickle_df_path, pickle_name=pickle_name)

if __name__ == '__main__':
    main()