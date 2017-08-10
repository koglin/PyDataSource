
def initArgs():
    """Initialize argparse arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attr", help='Input')
    parser.add_argument("option", nargs='?', default=None,
                        help='Optional attribute')
    parser.add_argument("-b", "--build", type=str,
                        help='Build html report')
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
    parser.add_argument("-M", "--max_size", type=int,
                        help='Maximum data array size')
    #parser.add_argument("--make_summary", action="store_true", default=False,
    #                    help='Make summary for array data.')
    return parser.parse_args()


