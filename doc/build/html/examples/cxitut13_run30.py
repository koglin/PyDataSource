import pandas as pd
import PyDataSource
ds = PyDataSource.DataSource(exp='cxitut13',run=30)
evt = ds.events.next()
evt.Sc2Imp.add.module('impbox')
pd.DataFrame(evt.Sc2Imp.waveform.T).plot()
pd.DataFrame(evt.Sc2Imp.filtered.T).plot()

