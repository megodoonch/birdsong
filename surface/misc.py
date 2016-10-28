import numpy as np

# Generate some n number of colours, hopefully maximally different


# source: http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
import colorsys

def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (20 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors




def cohens_d(x, y):
    lx = len(x)- 1
    ly = len(y)- 1
    md  = abs(np.mean(x) - np.mean(y))        ## mean difference (numerator)
    csd = lx * np.var(x) + ly * np.var(y)
    csd = csd/(lx + ly)
    csd = np.sqrt(csd)                        ## common sd computation
    cd  = md/csd                              ## cohen's d
    return cd







def get_freqs(lst):
    # Return a list of pairs that correspond to counts of 
    # elements: (a,n) means that a appeared n times in the list.
    # The list is ordered by a, whatever order that variable has.
    counts = {}
    for l in lst:
        counts[l] = counts.get(l,0)+1

    # Convert to a list of pairs
    pairs = counts.items()

    # Order by the first element of each pair
    # pairs = pairs.sort(cmp=lambda (x,na),(y,nb): cmp(x,y))

    return pairs
