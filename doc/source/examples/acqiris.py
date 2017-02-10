import PyDataSource

ds = PyDataSource.DataSource(exp='amotut13',run=206)
evt  = ds.events.next()
evt.AmoETOF_0_Acqiris_0.show_info()
evt.AmoETOF_0_Acqiris_0.waveform.shape

import matplotlib.pyplot as plt
plt.plot(evt.AmoETOF_0_Acqiris_0.wftime[0], evt.AmoETOF_0_Acqiris_0.waveform[0])
plt.show()

