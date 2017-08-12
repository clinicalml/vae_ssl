import time
import os
import sys
from contextlib import contextmanager
import datetime
import json
import pandas as pd

def extract_configs(rootdir,filename='config.json'):
    files = {}
    for dirpath, dirnames, filenames in os.walk(rootdir):
        if filename in filenames:
            with open(os.path.join(dirpath,filename),'r') as f:
                files[dirpath] = json.loads(f.read())
    return pd.DataFrame(files)

def get_new_dirpath_as_timestring(parentdir,suffix=None,leading_zeros_counter=2):
    assert isinstance(leading_zeros_counter,int), 'leading_zeros_counter must be an int'
    assert leading_zeros_counter > 0, 'leading_zeros_counter must be greater than 0'
    time_str = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
    i=0
    pathexists = True
    while pathexists:
        # e.g. if leading_zeros_counter=3, counter_str = '%03d'
        counter_str = '_%' + '0%dd' % leading_zeros_counter
        dirname = time_str 
        if suffix is not None:
            dirname += '_%s' % suffix
        if i > 0:
            dirname += counter_str % i
        pathexists = os.path.exists(os.path.join(parentdir,dirname))
        i+=1
        assert i < 10**leading_zeros_counter, 'counter exceeded 10**leading_zeros_counter-1'
    return os.path.join(parentdir,dirname)

def format_timestr(seconds):
    hours = int(seconds/3600)
    minutes = int((seconds % 3600)/60)
    seconds = int(seconds - hours*3600 - minutes*60)
    return '%dH:%dM:%dS' % (hours,minutes,seconds)
    

@contextmanager
def track_and_print_time(start_str=None,end_str=None):
    start_time = time.time()
    if start_str:
        print start_str
    yield
    duration = datetime.timedelta(seconds=time.time()-start_time)
    print_str = ''
    if end_str:
        print_str += end_str + ' '
    elif start_str:
        print_str += 'finished: ' + start_str + ' '
    print_str += '(duration: %d days, %s)' % (duration.days,format_timestr(duration.seconds))
    print print_str


class Logger(object):
    def __init__(self,logfile='logfile.log',mode='w'):
        self.logfilepath = logfile
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log = open(logfile,mode)

    @contextmanager
    def send_stdout_to_logger(self):
        sys.stdout = self
        yield
        sys.stdout = self.stdout

    @contextmanager
    def send_stderr_to_logger(self):
        sys.stderr = self
        yield
        sys.stderr = self.stderr
        

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)  

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def isatty(self):
        return self.stdout.isatty()


if __name__=='__main__':
    import sys
    logger = Logger()

    with logger.send_stderr_to_logger():
        with logger.send_stdout_to_logger():
            
            #test track_and_print_time 
            with track_and_print_time('start stuff','end stuff'):
                #test get_new_dirpath_as_timestring
                test_dir = '/tmp/test_get_new_dirpath_as_timestring/'
                os.system('mkdir -p %s' % test_dir)    
                os.system('rm -rf %s/*' % test_dir)
                for i in range(10):
                    dirpath = get_new_dirpath_as_timestring(test_dir,None,2)
                    os.system('mkdir -p %s' % dirpath) 
                    print dirpath
                for i in range(10):
                    dirpath = get_new_dirpath_as_timestring(test_dir,'suffix',2)
                    os.system('mkdir -p %s' % dirpath) 
                    print dirpath
                print format_timestr(1234567.2354)
                time.sleep(1)

            with track_and_print_time('just start stuff'):
                time.sleep(1)

            with track_and_print_time():
                print 'not starting any stuff'
                time.sleep(1)

            with track_and_print_time(end_str='only end string'):
                print 'only an end string'
                time.sleep(1)

            from progressbar import ProgressBar
            import time
            import numpy as np
            n = 30

            with ProgressBar(n,prefix='progress bar') as progressbar:
                for i in range(n):
                    time.sleep(0.1)
                    extra_str = ['extra %s' % (np.random.randn()) for j in range(n-i+3)]
                    progressbar.update(i+1,extra_str)
                    if i==10 and False:
                        assert False, 'raising an error'
                output_str = progressbar.get_current_output_str()
            print output_str
                               

            print len(extra_str)

                    

