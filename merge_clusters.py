import os
import glob
import pandas as pd

pkllist = glob.glob( "cluster-pkl/*.pkl" )
df_temp_reconstructed = pd.concat( [ pd.read_pickle( pkllist[i], compression='bz2' ) for i in range( len( pkllist ) ) ])
df_temp_reconstructed = df_temp_reconstructed.sort_values(['stationcode','year'], ascending=[True, True]).reset_index(drop=True)
df_temp_reconstructed.to_pickle( 'df_temp_reconstructed.pkl', compression='bz2' ) 


