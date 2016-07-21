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
psana_attrs = {a: {} for a in dir(psana) if not a.startswith('_') \
              and not a.startswith('ndarray') and a not in psana_omit_list}

for mod_name in psana_doc_info:
    mod = getattr(psana,mod_name)
    psana_doc_info[mod_name] = {a: {} for a in dir(mod) if not a.startswith('_')}
    for typ_name in psana_doc_info[mod_name]:
        typ = getattr(mod, typ_name)
        psana_doc_info[mod_name][typ_name] = {a: {} for a in dir(typ) if not a.startswith('_') }
        # Default convention is that attributes have lower case.  
        # Configure exceptions below (e.g. Generic1D)
        psana_attrs[mod_name][typ_name] = [a for a in dir(typ) if not a.startswith('_') and not a[0].isupper() ]
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
psana_doc_info['Acqiris']['ConfigV1']['channelMask']['func_method'] = bin 
psana_doc_info['Acqiris']['HorizV1']['sampInterval']['unit'] = 'sec'
psana_doc_info['Acqiris']['HorizV1']['delayTime']['unit'] = 'sec'
psana_doc_info['Camera']['FrameFexConfigV1']['threshold']['unit'] = ''
psana_doc_info['Quartz']['ConfigV2']['gain_percent']['unit'] = ''
psana_doc_info['Quartz']['ConfigV2']['max_taps']['unit'] = ''
psana_doc_info['Quartz']['ConfigV2']['output_resolution']['unit'] = ''

psana_doc_info['Generic1D']['DataV0']['data_u16']['func_shape'] = 8
psana_doc_info['Generic1D']['DataV0']['data_u32']['func0'] = 8     # offset
psana_doc_info['Generic1D']['DataV0']['data_u32']['func_shape'] = 8

psana_attrs['Generic1D']['ConfigV0'] = [
                                         'Depth',
                                         'Length',
                                         'NChannels',
                                         'Offset',
                                         'Period',
                                         'SampleType',
                                         'Sample_Type',
                                         'data_offset']


# Common mode not applicable?
#psana_doc_info['CsPad']['ElementV2']['common_mode']['func_quads'] = 'quads_shape'
psana_doc_info['Acqiris']['DataDescV1Elem']['timestamp']['func_len'] = 'nbrSegments'
psana_doc_info['Acqiris']['DataDescV1']['data']['func_shape'] = 'data_shape'
psana_doc_info['Acqiris']['ConfigV1']['vert']['list_len'] = 'nbrChannels'

psana_doc_info['CsPad']['DataV1']['quads']['func_shape'] = 'quads_shape'
#psana_doc_info['CsPad']['DataV1']['quads']['func_dict_len'] = 'quads_shape'
psana_doc_info['CsPad']['DataV2']['quads']['func_shape'] = 'quads_shape'
#psana_doc_info['CsPad']['DataV2']['quads']['func_dict_len'] = 'quads_shape'
psana_doc_info['CsPad']['ConfigV3']['quads']['func_shape'] = 'quads_shape'
#psana_doc_info['CsPad']['ConfigV3']['quads']['func_dict_len'] = 'quads_shape'
psana_doc_info['CsPad']['ConfigV3']['numAsicsStored']['func_len'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV3']['roiMask']['func_len_hex'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV3']['roiMasks']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV4']['quads']['func_shape'] = 'quads_shape'
#psana_doc_info['CsPad']['ConfigV4']['quads']['func_dict_len'] = 'quads_shape'
psana_doc_info['CsPad']['ConfigV4']['numAsicsStored']['func_len'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV4']['roiMask']['func_len_hex'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV4']['roiMasks']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV5']['quads']['func_shape'] = 'quads_shape'
#psana_doc_info['CsPad']['ConfigV5']['quads']['func_dict_len'] = 'quads_shape'

# see https://confluence.slac.stanford.edu/display/PCDS/Discussion+of+timing+the+cspad+variants

psana_doc_info['CsPad']['ConfigV5']['numAsicsStored']['func_len'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV5']['asicMask']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV5']['badAsicMask0']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV5']['badAsicMask1']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV5']['concentratorVersion']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV5']['quadMask']['func_method'] = bin
psana_doc_info['CsPad']['ConfigV5']['roiMask']['func_len_hex'] = 'numQuads'
psana_doc_info['CsPad']['ConfigV5']['roiMasks']['func_method'] = hex
psana_doc_info['CsPad']['ElementV2']['common_mode']['func_shape'] = 32
psana_doc_info['CsPad']['ConfigV3QuadReg']['ampIdle']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV3QuadReg']['biasTuning']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV3QuadReg']['digCount']['func_method'] = hex
psana_doc_info['CsPad']['ConfigV3QuadReg']['acqDelay']['unit'] = 'x8ns'
psana_doc_info['CsPad']['ConfigV3QuadReg']['acqDelay']['doc'] = 'delay before acquisition (350 typical)'
psana_doc_info['CsPad']['ConfigV3QuadReg']['digDelay']['unit'] = 'x8ns'
psana_doc_info['CsPad']['ConfigV3QuadReg']['digDelay']['doc'] = 'hold delay before A to D conversion (1000 typical)'
psana_doc_info['CsPad']['ConfigV3QuadReg']['digPeriod']['unit'] = 'x8ns'
psana_doc_info['CsPad']['ConfigV3QuadReg']['digPeriod']['doc'] = 'digitiztion perios'
psana_doc_info['CsPad']['ConfigV3QuadReg']['intTime']['unit'] = 'x8ns'
psana_doc_info['CsPad']['ConfigV3QuadReg']['intTime']['doc'] = 'duration of the integration window (5000 typical)'
psana_doc_info['CsPad']['ConfigV3QuadReg']['readClkHold']['doc'] = '(should be 1)'
psana_doc_info['CsPad']['ConfigV3QuadReg']['readClkSet']['doc'] = '(should be 2)'
psana_doc_info['CsPad']['ConfigV3QuadReg']['rowColShiftPer']['doc'] = '(should be 3)'
psana_doc_info['CsPad']['ConfigV3QuadReg']['digCount']['doc'] = '(max = 0x3ff)'
psana_doc_info['CsPad']['CsPadReadOnlyCfg']['version']['func_method'] = hex

psana_doc_info['CsPad2x2']['ConfigV2']['concentratorVersion']['func_method'] = hex
psana_doc_info['CsPad2x2']['ConfigV2']['asicMask']['func_method'] = hex
psana_doc_info['CsPad2x2']['ConfigV2']['badAsicMask']['func_method'] = hex
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['pdpmndnmBalance']['doc'] = '2 bits per nibble, bit order pd00pm00nd00nm'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['pdpmndnmBalance']['unit'] = ''
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['acqDelay']['unit'] = 'x8ns'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['acqDelay']['doc'] = 'delay before acquisition (280 typical)'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['digDelay']['unit'] = 'x8ns'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['digDelay']['doc'] = 'hold delay before A to D conversion (960 typical)'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['digPeriod']['unit'] = 'x8ns'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['digPeriod']['doc'] = 'digitiztion perios'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['intTime']['unit'] = 'x8ns'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['intTime']['doc'] = 'duration of the integration window (5000 typical)'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['readClkHold']['doc'] = '(should be 1)'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['readClkSet']['doc'] = '(should be 1)'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['rowColShiftPer']['doc'] = '(should be 3)'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['digCount']['doc'] = '(max = 0x3ff)'
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['ampIdle']['func_method'] = hex
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['biasTuning']['func_method'] = hex
psana_doc_info['CsPad2x2']['ConfigV2QuadReg']['digCount']['func_method'] = hex
psana_doc_info['CsPad2x2']['CsPad2x2ReadOnlyCfg']['version']['func_method'] = hex
psana_doc_info['CsPad2x2']['CsPad2x2ReadOnlyCfg']['shiftTest']['func_method'] = hex
psana_doc_info['CsPad2x2']['ElementV1']['common_mode']['func_shape'] = 2

psana_doc_info['UsdUsb']['FexConfigV1']['name']['func_shape'] = 4

psana_doc_info['Ipimb']['ConfigV1']['capacitorValue']['func_index'] = 'capacitorValues'
psana_doc_info['Ipimb']['ConfigV2']['capacitorValue']['func_index'] = 'capacitorValues'
#Need to understand the diode scale and base arrays.
#psana_doc_info['Ipimb']['ConfigV2']['diode']['func_len'] = 4


