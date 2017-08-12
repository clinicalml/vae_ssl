import sys
import time
import reprint

# see https://github.com/Yinzo/reprint

class ProgressBar(reprint.output,object):
    
    def __init__(self,n,prefix=None,toolbar_width=30,output_to_stdout_only=True):
        self.n = n
        self.toolbar_width = toolbar_width
        self.prefix = prefix
        self.start_time = time.time()
        self._closed=False
        self.output_to_stdout_only = output_to_stdout_only #if true, only print to sys.__stdout__
        self.__stdout__ = sys.stdout
        self.reset_to_orig_stdout()
        super(ProgressBar,self).__init__()
        self.update(0)
        self.reset_to_curr_stdout()

    def reset_to_orig_stdout(self):
        if self.output_to_stdout_only:
            sys.stdout = sys.__stdout__

    def reset_to_curr_stdout(self):
        if self.output_to_stdout_only:
            sys.stdout = self.__stdout__
        
    def __enter__(self):
        self.reset_to_orig_stdout()
        return self

    def __exit__(self,*args,**kwargs):
        rval = super(ProgressBar,self).__exit__(*args,**kwargs)
        self.reset_to_curr_stdout()
        return rval

    def _set(self,i,val):
        if len(self.warped_obj)<=i:
            self.warped_obj.append(val)
        else:
            self.warped_obj[i] = val

    def get_current_output_str(self):
        return '\n'.join(self.warped_obj)
        
    def update(self,i,extra_str=None):
        dashes_per_i = float(self.toolbar_width)/float(self.n)
        duration = time.time() - self.start_time
        duration_per_i = 0 if i==0 else duration/float(i)
        estimated_total_time = None if i==0 else self.n*duration_per_i

        if estimated_total_time is None:
            timestr = ""
        else:
            duration_hrs = int(duration/3600)
            duration_diff = duration-duration_hrs*3600
            duration_min = int(duration_diff/60)
            duration_diff = duration_diff - duration_min*60
            duration_sec = int(duration_diff/1)
            total_hrs = int(estimated_total_time/3600)
            total_diff = estimated_total_time-total_hrs*3600
            total_min = int(total_diff/60)
            total_diff = total_diff - total_min*60
            total_sec = int(total_diff)

            if estimated_total_time > 3600:
                timestr = "%s hrs %s min / %s hrs %s min" % (duration_hrs,duration_min,total_hrs,total_min) 
            else:
                timestr = "%s min %s sec / %s min %s sec" % (duration_min,duration_sec,total_min,total_sec) 
            
            
        if self.prefix is not None:
            s = '%s [' % self.prefix
        else:
            s = '['
        s = '%s%s' % (s,'-'*int(i*dashes_per_i))
        s = '%s%s' % (s,' '*int((self.n-i)*dashes_per_i))
        s = '%s %s/%s' % (s,i,self.n)
        s = '%s %s' % (s,']')
        s = '%s ( %s )' % (s,timestr)
        #self.warped_obj[0] = s
        self._set(0,s)
        counter = 1
        if extra_str is not None:
            if isinstance(extra_str,str):
                self._set(1,extra_str)
                counter+=1
            elif isinstance(extra_str,list):
                for i,e in enumerate(extra_str):
                    self._set(i+1,str(e))
                    counter+=1
            else:
                self._set(1,str(extra_str))
                counter+=1
        for i in range(counter,len(self.warped_obj)):
            self._set(i,'')

    def close(self):
        pass

if __name__ == '__main__':
    import time
    import numpy as np
    n = 30
    with open('delete.me','w') as f:
        sys.stdout = f
        with ProgressBar(n,prefix='progress bar') as progressbar:
            for i in range(n):
                time.sleep(0.1)
                extra_str = ['extra %s' % (np.random.randn()) for j in range(n-i+3)]
                progressbar.update(i+1,extra_str)

    print len(extra_str)

