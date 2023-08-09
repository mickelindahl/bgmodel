"""
parameter_search.py - search parameter spaces.

$Id: parameter_search.py 373 2009-02-06 16:11:23Z mschmucker $

Author: Michael Schmuker (m.schmuker@fu-berlin.de)

parameter_search provides classes to calculate an arbitrary function's return
value for a list of input parameter sets. Most useful for "grid search", i.e.
searching the parameter space of a function by sampling it in a regular fashion.

Requirements:
- IPython >= 0.9 for parallel computing. 
- MPI (openmpi recommended) for distributed parallel computing
- mpi4py

General usage:
1.  Instantiate the ParameterSearcher class with a function and a list of
    dicitonaries. The function should take its parameters as a dictionary.
2.  Call search() on the instance. The provided function is called
    once for each dictionary in the list.
3.  Call harvest() on the instance. This returns a list of return values from
    each function call.

There are several classes available, which build on each other:

- ParameterSearcher: 
    Plain vanilla parameter searching: Takes a list of dictionaries and calls
    the provided function once for each dicitonary. 
- IPythonParameterSearcher:
    Uses IPython to parallel parameter searching. Function calls are executed
    in parallel on ipengine clients. Requires that the IPython controller and
    clients are set up.
- RestartingIPythonParameterSearcher: 
    Same as IPythonParameterSearcher, but restarts client engines for each
    evaluation of the provided function. Used to work around pyNN.neuron's 
    problems with resetting time to zero, or restarting simulation runs. 
    Requires the IPython controller to be set up.
- MPIRestartingIPythonParameterSearcher: 
    Same as RestartingIPythonParameterSearcher, but uses MPI to (re)start 
    client engines. Useful for distributed parallel computing. Requires the
    IPython controller to be set up.

Look at parameter_search_example.py and test_parameter_search.py for usage 
examples.

Todo:
- provide utility functions to start/stop controllers and engines.
- control logging of engines instead of just letting them dump to the console.
- extend to non-grid search algorithms (e.g. evo, swarm), possibly using 
  scipy.optimize.
"""

from NeuroTools import check_dependency

# Check availability and version of IPython
if check_dependency('IPython'):
    import IPython
    v = IPython.__version__.split('.')
    if float(v[2]) < 1. or float(v[1]) < 9.:
        print("""
----------------- Dependency Warning ---------------------
Warning: IPython must be version 0.9.1 or higher. 
         Parallel searching will most likely not work.""")
    try:
        import IPython.kernel.client
    except ImportError:
        print("""
----------------- Dependency Warning ---------------------
Warning: IPython seems to be installed without distributed 
         parallelization support.
         Parallel searching will not work.""")


class ParameterSearcher(object):
    """
    ParameterSearcher calls a function several times with varying arguments.
    The arguments are provided as a list of dictionaries, hence the function 
    must take a dictionary as only parameter. The function is called once for
    every element in the list of dictionaries. The result of the function call 
    is stored in a list.
    
    Using this class consists of three phases: 
    1. Init with argument list and function reference:
        ps = ParameterSearcher([{'arg1':1},{'arg1':2}], myfunc)
    2. performing the search:
        ps.search()
    3. retrieving the results:
        outlist = ps.harvest()
    """
    def __init__(self, dict_iterable = {}, func = None):
        """
        Create a ParameterSearcher object.
        Parameters:
        dict_iterable - list of parameter dictionaries
        func - the function to be executed with each element
               of dict_iterable as argument.
        """
        self.dict_iterable = dict_iterable
        self.func = func
        self.have_searched = False
        
    def search(self):
        """
        Perform parameter search. Only sanity checks are implemented here, 
        the actual searching is done by self._search(func,dict_iterable).
        The results are stored in self.outlist.
        """
        func = self.func
        dict_iterable = self.dict_iterable
        self._search(func, dict_iterable)
        self.have_searched = True
        
    def _search(self, func, dict_iterable):
        """
        Callback to actually do the searching. Calls the function provided in 
        the constructor with each element in the provided dict_iterable.
        When inheriting from ParameterSearcher, overwrite this function, not 
        search().
        
        Returns:
        outlist - list of output dictionaries
        """
        outlist = []
        for d in dict_iterable:
            ret = func(d)
            outlist.append(ret)
        self.outlist = outlist

    def harvest(self):
        """
        returns the list of outputs generated by the parameter search.
        """
        if not self.have_searched:
            raise(Exception('Must perform search before harvest.'))
        else: return self._harvest()

    def _harvest(self):
        """
        callback for harvesting the results.
        """
        return self.outlist
                  
class IPythonParameterSearcher(ParameterSearcher):
    """
    Uses IPython for parallel parameter searching. 
    """
    def __init__(self, dict_iterable = {}, func = None,
                 task_furl = None,
                 multiengine_furl = None,
                 engine_furl = None):
        """
        Sets the function to be called and the list of parameter dictinaries,
        connects to the IPython controller, distributes the tasks to the 
        engines and collects the results.
        
        Requires that ipcontroller and ipengine(s) are set up. If no FURLs are 
        given, the default location from the ipython setup is used.
        
        Parameters:
        dict_iterable - list of parameter dictionaries
        func - function to call with parameter dictionaries
        task_furl - FURL for task clients to connect to. 
        multiengine_furl - FURL for mltiengine clients to connect to
        engine_furl - FURL for ipengines to connect to
        """
        ParameterSearcher.__init__(self, dict_iterable, func)
        self.task_furl = task_furl
        self.multiengine_furl = multiengine_furl
        self.engine_furl = engine_furl
        from IPython.kernel import client
        self.mec = client.MultiEngineClient(furl_or_file = multiengine_furl)
        self.tc = client.TaskClient(furl_or_file = task_furl)

        # know which tasks we'll have to retrieve
        self.taskids = []
        # we keep track of failed tasks
        self.failed_tasks = []
    
    @staticmethod
    def make_controller(controller_command = 'ipcontroller', furl_dir = None, 
                        max_wait = 10.):
        """
        Start an ipcontroller. 
        
        Parameters: 
        controller_command - path to the command to invoke the controller with. 
                             Default requires the controller to be in the path.
        furl_dir - the directory to create furls in. Default is to create them 
                   in the system's default temp directory as returned by 
                   tempfile.gettempdir().
        max_wait - maximum number of seconds to wait for the controller to 
                   become accessible. It is polled three times a second during 
                   that time. 
                    
        Returns:Dictionary with keys:
        contr_obj - the controller's Popen-object 
        task_furl - path to the FURL for task clients
        multiengine_furl - path to the FURL for multiengine clients
        engine_furl - path to the FURL for engines
        """
        import subprocess, tempfile
        if furl_dir is None:
            furl_dir = tempfile.gettempdir()
        (fd, engine_furl) = tempfile.mkstemp(dir = furl_dir, 
                                             prefix = 'furl_engine_')
        (fd, multiengine_furl) = tempfile.mkstemp(dir = furl_dir,
                                                  prefix = 'furl_multiengine_')
        (fd, task_furl) = tempfile.mkstemp(dir = furl_dir,
                                           prefix = 'furl_task_')
        contr = subprocess.Popen(args = [controller_command, 
                              '--engine-furl-file=%s'%engine_furl,
                              '--multiengine-furl-file=%s'%multiengine_furl,
                              '--task-furl-file=%s'%task_furl])
        # wait until controller is accessible
        import time
        t = time.time()
        from IPython.kernel import client
        while True:
            try:
                mec = client.MultiEngineClient(furl_or_file = multiengine_furl)
                time.sleep(0.5)
                break
            except Exception, e:
                if (time.time() - t) < max_wait:
                    print "can't connect to controller yet. Retrying..."
                    time.sleep(0.33)
                else:
                    print "No connection after %f seconds. Giving up..."
                    raise e
        return {'contr_obj':contr, 
                'task_furl':task_furl, 
                'multiengine_furl':multiengine_furl,
                'engine_furl':engine_furl}
    
    def _search(self, func, dict_iterable):
        """
        Performs parameter search on IPython engines.
        """
        tasklist = self._prepare_tasklist(dict_iterable, func)
        self._work_tasklist(tasklist)

    def _prepare_tasklist(self, dict_iterable, func):
        """
        prepares a task list from the list of dictionaries. For each task, a 
        recovery function
        """
        from IPython.kernel import client
        tasklist = []
        for d in dict_iterable:
#            if recover:
#                def recov_func(func, params):
#                    try:
#                        return func(params)
#                    except Exception, e:
#                        return e, params
#                recov_task = client.MapTask(recov_func, [d, func])
#            if recover:
#                def recov_func(params):
#                    import os
#                    hostname = 
#                    result = subprocess.Popen(["hostname"], 
#                                              stdout=subprocess.PIPE).communicate()[0]
#                recov_task = client.MapTask("""""",
#                                     pull = "result")
#            else:
#                recov_task = None
            tasklist.append(client.MapTask(func, args = [d]) )
#            tasklist.append(client.MapTask("result = func(params)", 
#                            push = {'params':d},
#                            pull = "result",
#                            recovery_task = recov_task))
        return tasklist

    def _work_tasklist(self,tasklist):
        """
        performs the actual computation on the tasklist.
        """
        tc = self.tc
        taskids = [ tc.run(t) for t in tasklist ]
        tc.barrier(taskids)
        for ti in taskids:
            self.taskids.append(ti)

    def _upload_function(self,func):
        """
        uploads the function to be executed to the engines.
        """
        mec = self.mec
        import twisted.internet.error as tierror
        try:        
            mec.push_function(dict(func=func))
        except(tierror.ConnectionRefusedError):
            raise(Exception('Got ConnectionRefusedError. ' +
                            'Are ipcontroller and ipengines running?'))
                
    def _harvest(self):
        """
        Collect the results from the task client.
        """
        tc = self.tc
        status = tc.queue_status(verbose=True)
        # store failed tasks in Searcher to allow later access
        results = []
        for t in self.taskids:
            try:
                res = tc.get_task_result(t)
                results.append(res)
            except Exception, e:
                self.failed_tasks.append({'taskid':t, 'exception':e})
        return results


class RestartingIPythonParameterSearcher(IPythonParameterSearcher):
    """
    Uses IPython for parallel parameter searching, restarting the engines after
    a task has been finished. 
    
    Use this class (or MPIRestartingIPythonParameterSearcher below) to work 
    around problems that occur when starting several simulations in sequence 
    with pyNN.neuron.
    """
    def __init__(self, dict_iterable = {}, func = None,
                 task_furl = None,
                 multiengine_furl = None,
                 engine_furl = None,
                 numengines = 2, take_down = False,
                 enginecommand = 'ipengine'):
        """
        Sets up the IPython controllers, takes down any existing ipengines and 
        starts the defined number of ipengines.
        
        Parameters:
        dict_iterable - list of parameter dictionaries
        func - function to call with parameter dictionaries
        task_furl - FURL for task clients to connect to. 
        multiengine_furl - FURL for mltiengine clients to connect to
        engine_furl - FURL for ipengines to connect to
        take_down - kill existing ipengines. If False, raises Exception if 
                    running ipengines are encountered.
        enginecommand - command to use for launching ipengines
        """
        IPythonParameterSearcher.__init__(self, dict_iterable, func,
                                          task_furl = task_furl,
                                          multiengine_furl = multiengine_furl,
                                          engine_furl = engine_furl)
        self.numengines = numengines
        self.enginecommand = enginecommand
        self.results = []
        #are there any existing ipengines?
        mec = self.mec
        if (len(mec.get_ids()) != 0):
            if (take_down == False):
                raise(Exception('There are running ipengines.' +
                                'Refusing to take them down'))
            else: # kill existing ipengines
                mec.kill(controller=False, block=True)
        
    def _search(self, func, dict_iterable):
        tasklist = self._prepare_tasklist(dict_iterable, func)
        outlist = []
        while len(tasklist) > 0:
            tl_grouped = []
            for i in range(0,self.numengines):
                try:
                    tl_grouped.append(tasklist.pop(0))
                except(IndexError):
                    break
            self._execute_task_slice(tl_grouped)

    def _start_engines(self):
        """
        starts ipengines as subprocess.Popen objects and returns an array with 
        these objects.
        """
        engines = []
        import subprocess
        # gentlemen, start your engines
        for i in range(0,self.numengines):
            args = [self.enginecommand]
            if self.engine_furl is not None:
                args.append('--furl-file=%s'%self.engine_furl)
            engines.append(subprocess.Popen(args = args))
        self._wait_for_engines()
        return engines

    def _wait_for_engines(self):
        """
        wait for ipengines to come up.
        """
        import time, logging
        mec = self.mec
        numengines = self.numengines
        while len(mec.get_ids()) != numengines:
            logging.info('waiting a little more for engines to come up...')
            time.sleep(0.33)
        return
        
    def _execute_task_slice(self, tasks):
        """
        starts as many ipengines as tasks provided, executes the tasks, gets the
        results, takes the engines down again and returns the results.
        """
        engines = self._start_engines()
        # perform your tasks
        self._work_tasklist(tasks)
        self._stop_engines(engines)
        
    def _stop_engines(self, engines):
        """
        kills ipengines.
        """
        mec = self.mec
        # mec.kill(controller=False, block=True)
        # need to do this the hard way in order to make the IPython shell quit
        mec.execute('import os')
        import IPython.kernel.error
        try:
            mec.execute('os.kill(os.getpid(), 2)')
        except(IPython.kernel.error.CompositeError):
            pass #ignore the exception that is thrown here
        # wait for engines to come down
        import time
        while True:
            try:
                numeng = len(mec.get_ids())
                if numeng == 0:
                    break
            except Exception, e:
                print e
            finally:
                time.sleep(0.33)


class MPIRestartingIPythonParameterSearcher(RestartingIPythonParameterSearcher):
    """
    Uses IPython for parallel parameter searching, restarting the engines after
    a task has been finished. Uses MPI for starting engines in order to enable 
    parallel execution on a cluster.
    """
    def __init__(self, dict_iterable = {}, func = None,
                 task_furl = None,
                 multiengine_furl = None,
                 engine_furl = None,
                 numengines = 2, take_down = False,
                 enginecommand = 'ipengine',
                 mpirun_command='mpirun'):
        """
        Sets up the IPython controllers, takes down any existing ipengines and starts
        the defined number of ipengines.
        
        Parameters:
        dict_iterable - list of parameter dictionaries
        func - function to call with parameter dictionaries
        controller_ip - IP address of the IPython controller
        me_controller_port - port for MultiEngine connections
        tc_controller_port - port for Task connections
        numengines - the number of ipengines to use
        take_down - kill existing ipengines. If False, raises Exception if 
                    running ipengines are encountered.
        enginecommand - command to use for launching ipengines
        mpirun_command - the mpirun command (complete path, with all options)
        """
        RestartingIPythonParameterSearcher.__init__(self,dict_iterable, func,
                                          task_furl = task_furl,
                                          multiengine_furl = multiengine_furl,
                                          engine_furl = engine_furl,
                                          numengines = numengines, 
                                          take_down = take_down, 
                                          enginecommand = enginecommand)
        self.mpirun_command = mpirun_command

    def _start_engines(self):
        """
        starts ipengines as subprocess.Popen objects and returns an array with 
        these objects.
        """
        engines = []
        mpicommand = self.mpirun_command
        arglist = []
        import types
        if type(mpicommand) is types.ListType:
            for a in mpicommand: arglist.append(a)
        else: arglist.append(mpicommand)
        arglist.append('-np')
        arglist.append(str(self.numengines))
        if type(self.enginecommand) is not types.ListType:
            enginecommand = [self.enginecommand]
        else: 
            enginecommand = self.enginecommand
        if self.engine_furl is not None:
            enginecommand.append('--furl-file=%s'%self.engine_furl)
        for a in enginecommand: arglist.append(a)
        import subprocess
        # gentlemen, start your engines
        engines.append(subprocess.Popen(args = arglist))
        self._wait_for_engines()
        return engines
