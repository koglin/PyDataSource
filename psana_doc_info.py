import psana

def get_unit_from_doc(doc):
    """Parse the unit from the doc string.
    """
    invalid_units = ['this', 'long', 'all', 'setup', 'given', 'a', 'the']
    try:
        usplit = doc.rsplit(' in ')
        if 'Value' in doc and 'converted to' in doc:
            unit = '{:}'.format(doc.rsplit('converted to ')[-1].rstrip('.'))
        elif len(usplit) < 2:
            unit = ''
        else:
            unit = '{:}'.format(usplit[-1])
            unit = unit.rstrip('.').rstrip(',').rsplit(' ')[0].rstrip('.').rstrip(',')
            
            if unit.endswith('(') or unit in invalid_units:
                unit = ''
        
    except:
        unit = ''
    return unit

def get_type_from_doc(doc):
    """Parse the type from the doc string.
    """
    try:
        return  doc.replace('\n',' ').split('-> ')[1].split(' ')[0]
    except:
        return None

# create dictionary of psana method doc and unit information
psana_omit_list = ['logging', 'os', 'setConfigFile', 'setOption', 'setOptions']
psana_doc_info = {a: {} for a in dir(psana) if not a.startswith('_') \
              and not a.startswith('ndarray') and a not in psana_omit_list}
for mod_name in psana_doc_info:
    mod = getattr(psana,mod_name)
    psana_doc_info[mod_name] = {a: {} for a in dir(mod) if not a.startswith('_')}
    for typ_name in psana_doc_info[mod_name]:
        typ = getattr(mod, typ_name)
        psana_doc_info[mod_name][typ_name] = {a: {} for a in dir(typ) if not a.startswith('_') }
        for attr in psana_doc_info[mod_name][typ_name]:
            if attr in ['TypeId','Version']:
                info = {'doc': '', 'unit': '', 'type': ''}
            else:
                func = getattr(typ, attr)
                doc = func.__doc__
                if doc:
                    doc = doc.split('\n')[-1].lstrip(' ')
                    if doc.startswith(attr):
                        doc = ''

                info = {'doc': doc, 
                        'unit': get_unit_from_doc(func.__doc__), 
                        'type': get_type_from_doc(func.__doc__)}
            
            psana_doc_info[mod_name][typ_name][attr] = info 

# Updates to psana_doc_info info
psana_doc_info['Bld']['BldDataEBeamV7']['ebeamDumpCharge']['unit'] = 'e-'
psana_doc_info['Bld']['BldDataFEEGasDetEnergyV1']['f_11_ENRC']['unit'] = 'mJ'
psana_doc_info['Bld']['BldDataFEEGasDetEnergyV1']['f_12_ENRC']['unit'] = 'mJ'
psana_doc_info['Bld']['BldDataFEEGasDetEnergyV1']['f_21_ENRC']['unit'] = 'mJ'
psana_doc_info['Bld']['BldDataFEEGasDetEnergyV1']['f_22_ENRC']['unit'] = 'mJ'
psana_doc_info['Bld']['BldDataFEEGasDetEnergyV1']['f_63_ENRC']['unit'] = 'mJ'
psana_doc_info['Bld']['BldDataFEEGasDetEnergyV1']['f_64_ENRC']['unit'] = 'mJ'
psana_doc_info['Acqiris']['DataDescV1Elem']['nbrSamplesInSeg']['unit'] = ''
psana_doc_info['Camera']['FrameFexConfigV1']['threshold']['unit'] = ''
psana_doc_info['Quartz']['ConfigV2']['gain_percent']['unit'] = ''
psana_doc_info['Quartz']['ConfigV2']['max_taps']['unit'] = ''
psana_doc_info['Quartz']['ConfigV2']['output_resolution']['unit'] = ''

# Common mode not applicable?
#psana_doc_info['CsPad']['ElementV2']['common_mode']['func_quads'] = 'quads_shape'
psana_doc_info['Acqiris']['DataDescV1Elem']['timestamp']['func_len'] = 'nbrSegments'
psana_doc_info['Acqiris']['DataDescV1']['data']['func_shape'] = 'data_shape'
psana_doc_info['CsPad']['ConfigV5']['quads']['func_shape'] = 'quads_shape'
psana_doc_info['CsPad']['DataV2']['quads']['func_shape'] = 'quads_shape'
psana_doc_info['CsPad']['ConfigV5']['numAsicsStored']['func_len'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV5']['roiMask']['func_len_hex'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV5']['roiMasks']['func_method'] = hex
psana_doc_info['Ipimb']['ConfigV2']['capacitorValue']['func_index'] = 'capacitorValues'

