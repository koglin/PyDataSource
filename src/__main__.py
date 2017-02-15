import PyDataSource
import time
import argparse

def initArgs():
    """Initialize argparse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("attr", help='Input')
    parser.add_argument("option", nargs='?', default=None,
                        help='Optional attribute')
    parser.add_argument("-e", "--exp", type=str,
                        help='Experiment')
    parser.add_argument("-r", "--run", type=str,
                        help='Run')
    parser.add_argument("-i", "--instrument", type=str,
                        help='Instrument')
    parser.add_argument("--nchunks", type=int,  
                        help='total number of chunks')
    parser.add_argument("--ichunk", type=int,  
                        help='chunk index')
    parser.add_argument("-c", "--config", type=str,
                        help='Config File')
    parser.add_argument("-s", "--station", type=int,
                        help='Station')
    parser.add_argument("-n", "--nevents", type=int,
                        help='Number of Events to analyze')
    #parser.add_argument("--make_summary", action="store_true", default=False,
    #                    help='Make summary for array data.')
    return parser.parse_args()


if __name__ == "__main__":
    time0 = time.time()
    args = initArgs()
    attr = args.attr
    exp = args.exp
    run = int(args.run)
    ds = PyDataSource.DataSource(exp=exp,run=run)
    if attr == 'config':
        print ds._get_config_file()
    if attr == 'configData':
        print ds.configData.show_info()
    if attr in ['steps','nsteps']:
        print ds.configData.ScanData.nsteps        
    if attr in ['events','nevents']:
        print ds.nevents        
    if attr in ['scan']:
        print ds.configData.ScanData.show_info()
    if attr in ['xarray']:
        print 'to_xarray'
        if args.config:
            print 'Loading config: {:}'.format(args.config)
            ds.load_config(file_name=args.config)
        
        print 'to_xarray...'
        x = ds.to_xarray(save=True, 
                         nevents=args.nevents, 
                         nchunks=args.nchunks, 
                         ichunk=args.ichunk)
        print x


    #print 'Total time = {:8.3f}'.format(time.time()-time0)

