import PyDataSource

a = PyDataSource.open_h5netcdf(exp='xpptut15', run=200, summary=True)

# Default uses 'lxt_vitara_ttc' as xaxis
PyDataSource.xy_ploterr(a, 'EBeam_ebeamPhotonEnergy')

PyDataSource.xy_ploterr(a, 'EBeam_ebeamPhotonEnergy', xaxis='EBeam_ebeamCharge')


