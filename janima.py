'''
Created by Joris Josiek
Email: joris.josiek@gmail.com
Last Edit: April 30, 2020

This programm aims to make python animations simpler, so that the user
will only have to call run and pass the f(x,t) functions as well as
some animation parameters. It handles all setup and framework for animation.
'''

import numpy as np
from matplotlib import pyplot as plt

# Styling
plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rc('text', usetex=True)

# Animation Module
from matplotlib.animation import FuncAnimation

class anima:
    def __init__(self, ts, xs, params={}, funcs=[]):
        '''
            xs : array of values for x if not specified for functions
            params : global parameters of animation
            funcs : list of dicts containing functions (x,t), subplot coords,
                        extra function parameters OR
                    single dict
                    single function
        '''
        self.t_values = ts
        self.frames = len(ts)
        self.x_values = xs
        self.parameters = {}
        
        self.set_parameters(params)
        self.set_functions(funcs)
        
    def set_parameters(self, params):
        '''
        Fills up passed global plot parameters with default parameters.
        To define new parameters, they must be added to the default_parameters 
        dict.
        ''' 
        
        # All dict objects here take subplot numbers as indices.
        default_params = {'suptitles' : {},
                          'axlimits' : {},
                          'axticks' : {},
                          'save' : False,
                          'show' : True}
        for parameter in default_params:
            if parameter in params:
                self.parameters[parameter] = params[parameter]
            else:
                self.parameters[parameter] = default_params[parameter]
        
    def set_functions(self, funcs):
        '''
        Takes passed functions and turns them into the correct format for
        the framework. Acceptable values for funcs:
            - a single function of two variables f(x,t)
            - a dict containing said function in 'func' with other parameters
                (incl. mandatory subplot number indexed by 'subplt')
            - a list contain dicts as in the previous point
        '''
        if type(funcs) == list:
            self.functions = funcs
        elif type(funcs) == dict:
            self.functions = [funcs]
            funcs = self.functions
        elif type(funcs) == type(lambda : 0):
            self.functions = [{'func':funcs, 'subplt':111}]
            funcs = self.functions
        self.set_funcparameters(funcs)
    
    def set_funcparameters(self, funcs):
        '''
        Takes functions dicts and appends necessary parameters. New parameters
        should be added to default_params before implementation, in order to
        avoid index errors.
        '''
        default_params = {'xs' : self.x_values,
                          'c' : 'auto',
                          'lw' : 'auto',
                          'marker' : ' '}
        
        # Fill undefined parameters with defaults
        for func in self.functions:
            if func in funcs:
                for parameter in default_params:
                    if not parameter in func:
                        func[parameter] = default_params[parameter]

    def run(self):

        # Extract unique list of subplot grids (xxx) as passed to subplot
        subplotGrids = list(set([func['subplt'] for func in self.functions]))
            
        # Dict with axes limits by subplt
        xmin, xmax = {}, {} 
        ymin, ymax = {}, {} 
    
        # Create subplot objects and save them in dictionary
        PlotHome = {}
        fig = plt.figure('Animator', figsize=(12, 8))
        fig.tight_layout()
    
        for subplt in subplotGrids:
            PlotHome[subplt] = fig.add_subplot(subplt)
            
            # Set formatting:
            
            # Dashed Axes
            PlotHome[subplt].axvline(x=0, ls='--', lw=1, c=(1, 1, 1, 0.1))
            PlotHome[subplt].axhline(y=0, ls='--', lw=1, c=(1, 1, 1, 0.1))
            
            # Implement subplot-specific parameters here
            if subplt in self.parameters['suptitles']: 
                PlotHome[subplt].set_title(
                        self.parameters['suptitles'][subplt], size=16)
            if subplt in self.parameters['axticks']:
                PlotHome[subplt].set_xticks(
                        self.parameters['axticks'][subplt][0])
                PlotHome[subplt].set_yticks(
                        self.parameters['axticks'][subplt][1])
            else:
                PlotHome[subplt].set_xticks([])
                PlotHome[subplt].set_yticks([])
            
            
            # Create empty array to later fill with extreme values for funcs
            xmin[subplt] = []
            xmax[subplt] = []
            ymin[subplt] = []
            ymax[subplt] = []
            
        # Set spacing between subplots (error in case of single row/column)
        try:
            plt.subplots_adjust(hspace=0.5)
        except:
            pass
        try:
            plt.subplots_adjust(vspace=0.5)
        except:
            pass
    
        # Create plots to animate
        lines = []
        for func in self.functions:
            # Implement function-specific parameters in this block.
            
            # Parameters that need to be passed directly to plt.plot
            plot_params = {p:func[p] for p in func 
                           if (p in ['c', 'lw'] and func[p]!='auto')}
            
            f = func['func']
            subplt = func['subplt']
            xs = func['xs']
            lines.append(PlotHome[subplt].plot([], [], **plot_params)[0])
            
            # Create arrays with _all_ y values (over all t)
            ys = np.array([])
            for t in self.t_values:
                ys = np.concatenate([ys, f(xs, t)])
                
            # Append extreme values to the field (used to set axes limits)
            xmin[subplt].append(min(xs))
            xmax[subplt].append(max(xs))
            ymin[subplt].append(min(ys))
            ymax[subplt].append(max(ys))
        
        # Choose extreme value of all functions in subplot as axis limits
        for subplt in PlotHome:
            minx, maxx = min(xmin[subplt]), max(xmax[subplt])
            miny, maxy = min(ymin[subplt]), max(ymax[subplt])
            dy = maxy - miny
            if subplt in self.parameters['axlimits']:
                axlimits = self.parameters['axlimits'][subplt]
                if not axlimits[0] is None:
                    PlotHome[subplt].set_xlim(axlimits[0])
                else:
                    PlotHome[subplt].set_xlim([minx, maxx])
                if not axlimits[1] is None:
                    PlotHome[subplt].set_ylim(axlimits[1])
                else:
                    PlotHome[subplt].set_ylim([miny-0.1*dy, maxy+0.1*dy])
            else:
                PlotHome[subplt].set_xlim([minx, maxx])
                PlotHome[subplt].set_ylim([miny-0.1*dy, maxy+0.1*dy])
        
        # Animation Framework   
            
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        def animate(i):
            for j, line in enumerate(lines):
                line.set_data(self.functions[j]['xs'], 
                              self.functions[j]['func'](
                                      self.functions[j]['xs'], 
                                      self.t_values[i]))
            return lines
        
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=self.frames,
                             interval=1,
                             blit=True,
                             save_count=self.frames,
                             repeat=True)
        
        # Saving requires imagemagick to be installed!
        if self.parameters['save']:
            anim.save('./janimation.gif', 
                      writer='imagemagick', 
                      fps=60, bitrate=-1)
        
        if self.parameters['show']:
            plt.show()
    

if __name__ == "__main__":
    # For testing purposes:
    
    def test_function1(x, t):
        return np.sin(4*x-t)
    
    def test_function2(x, t):
        return test_function1(x, t)*np.exp(-((x-t)/4)**2)
    
    xs = np.linspace(-10, 10, 1000)
    ts = np.linspace(-10*3.14, 10*3.14, 500)
    
    Animation1 = anima(ts, xs)
    
    Animation1.set_functions([{'func':test_function1, 
                               'subplt':211, 
                               'c':'yellow'}, 
                              {'func':test_function2, 
                               'subplt':212}])
    
    Animation1.set_parameters({'suptitles': {211:'Harmonic wave',
                                             212:'Modulated wave packet'}})
    
    Animation1.run()