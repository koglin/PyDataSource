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
    parser.add_argument("-s", "--station", type=int,
                        help='Station')
    return parser.parse_args()


if __name__ == "__main__":
    time0 = time.time()
    args = initArgs()
    attr = args.attr
    exp = args.exp
    run = args.run
    ds = PyDataSource.DataSource(exp=exp,run=run)
    if attr == 'config':
        print ds.configData.show_info()
    if attr in ['steps','nsteps']:
        print ds.configData.ScanData.nsteps        
    if attr in ['events','nevents']:
        print ds.nevents        
    if attr in ['scan']:
        print ds.configData.ScanData.show_info()

    #print 'Total time = {:8.3f}'.format(time.time()-time0)

