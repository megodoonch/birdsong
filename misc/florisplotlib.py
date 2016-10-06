
import numpy as np
import colorsys

def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (20 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors



import pandas as pd
from matplotlib.pyplot import *
import matplotlib
import seaborn as sns


def mypointplot(data,x,y,hue,col=None,row=None,dodge=0,title="",errval="se",dot=0,ylim=None,hue_colours="categorical"):
    """ Basically reproduce the point plot function but allowing x factor levels 
    to be numeric (so that we produce the proper spacing)
    
    x is the variable name mapped to the x axis
    y is the variable mapped to the y axis
    hue is the variable name that becomes encoded by the colour
    row, col are the variables mapped to rows and columns of the plot (i.e. there will be row x col plots in total)
    dot is the size of the dots (if we plot dots; 0 will not plot dots)
    errval encodes the error to be plotted: sd or se. None will not plot error bars at all.
    
    """

    assert hue_colours in ["categorical","continuous"]
    
    groupvals=[x,hue]
    ncol,nrow = 1,1
    colvalues,rowvalues=[None],[None]
    if col!=None:
        colvalues = list(set(data[col]))
        colvalues.sort()
        ncol = len(colvalues)
        groupvals+=[col]
    if row!=None:
        rowvalues = list(set(data[row]))
        rowvalues.sort()
        nrow = len(rowvalues)
        groupvals+=[row]
        
    groupvals = list(set(groupvals))
        
    #print(colvalues)
    #print(rowvalues)
    #print(groupvals)
    sels = data[ ~np.isnan(data[y]) & ~np.isinf(data[y]) ] # remove NANs

    mse = sels.groupby(groupvals)[y].agg({'N':len,'mean.%s'%y:np.mean,'sd.%s'%y:np.std}).reset_index()
    mse["se.%s"%y] = mse["sd.%s"%y]/np.sqrt(mse["N"])

    #print(sels)

    f,axarr = subplots(nrow,ncol,sharey=True,sharex=True,squeeze=False)
    
    huevalues = list(set(data[hue]))
    huevalues.sort()

    if hue_colours=="categorical":
        colors = get_colors(len(huevalues))
        hue_plot_colors = dict(zip(huevalues,colors))
    if hue_colours=="continuous":
        hue_plot_colors = dict([ (t,cm.jet(float(c)/len(huevalues))) for c,t in enumerate(huevalues)])

    
    #print(colvalues)
    for ic,c in enumerate(colvalues):
        #print(c)
        #print(colvalues)
        if col!=None: # select by column
            datc = mse[ mse[col]==c ]
            collab = "%s %s"%(col,c)
        else:
            datc = mse
            collab = ""        
        
        for ir,r in enumerate(rowvalues):
            #print(r)
            #print(rowvalues)
            if row!=None: # select by row
                datr = datc[ datc[row]==r ]
                rowlab = "%s %s"%(row,r)
            else:
                datr = datc
                rowlab = ""
            
            #print("Selecting %i, %i"%(ir,ic))
            ax = axarr[ir,ic]
            
            for i,h in enumerate(huevalues):
                toplot = datr[ datr[hue]==h ]
                if toplot.shape[0]!=0: # if there are actually rows
                    xvals = list(dodge*(float(i)/len(huevalues))+toplot[x])
                    yvals = list(toplot["mean.%s"%y])
                    if errval==None:
                        ax.plot(xvals,
                                yvals,
                                '-o',
                                clip_on=False,label=h,lw=2.5,markersize=dot,color=hue_plot_colors[h])
                    else:
                        errvals = list(toplot["%s.%s"%(errval,y)])
                        ax.errorbar(xvals,
                                    yvals,
                                    errvals,
                                    clip_on=False,label=h,lw=2.5,markersize=dot,color=hue_plot_colors[h])

            xmin,xmax=min(mse[x]),max(mse[x])
            xran=xmax-xmin
            ax.set_xlim(xmin-.05*xran,xmax+.05*xran)
            
            sns.despine(offset=5,ax=ax)
            ax.set_title(" ".join([collab,rowlab]))
    
    if hue_colours=="categorical":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title=hue)
    
    if ylim!=None:
        ax.set_ylim(ylim)
    
    if hue_colours=="continuous":
        #pass
        hue_positions   = [ float(c)/len(huevalues) for c,t in enumerate(huevalues)]
        #hue_plot_colors = dict([ (t,cm.jet(c))           for t,p in zip(huevalues,hue_positions)])

        f.subplots_adjust(right=0.85)
        cbar_ax = f.add_axes([0.9, 0, 0.1, 1])
        cb1 = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cm.jet,
                                               #norm = matplotlib.colors.Normalize(vmin=5, vmax=10),
                                               orientation='vertical')
        cb1.set_label(hue)
        #cbar = f.colorbar(ax=ax)
        cb1.ax.get_yaxis().set_ticks(hue_positions)
        cb1.ax.get_yaxis().set_ticklabels(huevalues)
        #for t,:
        #    cbar.
        #    cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
        #cbar.ax.get_yaxis().labelpad = 15
        #cbar.ax.set_ylabel(hue, rotation=270)

    
    #return mse
    return f,ax



def fisherz(r):
    return .5*np.log((1+r)/(1-r))
