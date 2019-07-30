__author__ = "Ryan H. Hsu"
__email__ = "ryan.hsu@berkeley.edu"
__version__ = "2.0"

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import itertools

def pull_image_roots(directory):
    fls = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        fls.extend(filenames)
        break
    
    roots = []
    
    for f in fls:
        if f.find(TAGS[0]):
            candidate_root = f[:f.find(TAGS[0])]
            fluor_tags = TAGS[1:]
            num_tags_found = 0
            for c_tag in fluor_tags:
                if candidate_root+c_tag in fls:
                    num_tags_found += 1
            
            if num_tags_found == len(fluor_tags):
                roots.append(candidate_root)
    
    fullpaths = [directory+'/'+r for r in roots]
    
    return roots, fullpaths

def printImage(im, umPerPixel=None, scalebarSize=100):
    plt.subplots(figsize=(10, 10))

    temp_im = im.copy()
    if umPerPixel:
        scaleWidth = scalebarSize/umPerPixel
        temp_im[0:5, 0:int(scaleWidth), 0] = 0
        temp_im[0:5, 0:int(scaleWidth), 1] = 1000
        temp_im[0:5, 0:int(scaleWidth), 2] = 1000
    
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(temp_im)
    plt.show()

def findCirclesFromBF(BF_img, diameter_val=144, threshold_val=30, display=False):

    gray = cv2.cvtColor(BF_img, cv2.COLOR_BGR2GRAY)
    BF_output = BF_img.copy()

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                      minDist=max(1,2*int((diameter_val/2))),
                      minRadius=int(0.9*(diameter_val/2)),
                      maxRadius=int(1.1*(diameter_val/2)),
                      param2=threshold_val)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            cv2.circle(BF_output, (x, y), r, (91, 244, 255), 2)

    if display:
        printImage(BF_output)
        
    return BF_output, circles

def loadAndScaleImage(f_name, scale=[0, 255], display=False, hist=False):

    '''scale factor can either be an int (in which case all pixels will be multiplied by scale)
    or can be a list of two ints, indicating the lower and upper bound cuttoffs for pixel intensities [0,255].
    Pixel intensities between the two values will linearly interpolated'''
    
    si = None

    raw_img = cv2.imread(f_name)[...,::-1]
    height, width, channels = raw_img.shape

    si = raw_img.astype('uint64').copy()

    if type(scale) == int:
        si *= scale
        si[si[:,:,:] > 255] = 255

    if type(scale) == list:
        x0, x1 = float(scale[0]), float(scale[1])
        y0, y1 = 0., 255.

        m = (y1-y0)/(x1-x0)
        b = y0 - m*x0
        
        def line_eq(x):
            return (m*x)+b
        v_line_eq = np.vectorize(line_eq)
        si = v_line_eq(si)
        si[si[:,:,:] > 255] = 255
        si[si[:,:,:] < 0] = 0

    si = si.astype('uint8')
    if display:
        printImage(si)
        
    if hist and type(scale) == list:
        plt.subplots(figsize=(15, 3))
        plt.title('Pixel Intensities')
        sns.distplot(raw_img[:,:,].flatten(), bins=256)
        highy = plt.gca().get_ylim()[1]
        plt.scatter([scale[0], scale[1]], [0, highy], color='red', s=20)
        plt.plot([scale[0], scale[1]], [0, highy], color='red')
        plt.xticks(np.arange(0,256,5))
        plt.xlim([-5, 255+5])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.show()
        
        sns.despine()
    return si

def drawCirclesOnImage(image, circles, color=(255, 255, 255)):
    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, color, 2)
        return image
    return None

def extract_circles_regions(image, circles):
    regions = []
    for c in circles:
        cx, cy, cr = c[0], c[1], c[2]
        if cx-cr>0 and cx+cr<image.shape[1] and cy-cr>0 and cy+cr<image.shape[0]:
            droplet_region = image[cy-cr: cy+cr, cx-cr:cx+cr]
            regions.append(droplet_region)
    return regions

def extractAllCircleRegions(fullpaths, DIAMETER_VAL, THRESHOLD_VAL, TAGS, SCALES):
    # Collect extracted circles here
    extracted_circles_dict = {}
    for tag in TAGS:
        extracted_circles_dict[tag] = []

    for i, path_root in enumerate(fullpaths):
        print str(i+1) +'/' + str(len(fullpaths))
        loaded_BF_img = loadAndScaleImage(path_root+BF_TAG, scale=BF_SCALE, display=False)
        circumscribed_BF_img, circles = findCirclesFromBF(loaded_BF_img,
                                                          diameter_val=DIAMETER_VAL,
                                                          threshold_val=THRESHOLD_VAL,
                                                          display=True)

        loaded_channels = []
        for tag, scale in zip(TAGS, SCALES):
            loaded_image = loadAndScaleImage(path_root+tag, scale=scale)
            extracted_circles = extract_circles_regions(loaded_image, circles)
            extracted_circles_dict[tag] = extracted_circles_dict[tag] + extracted_circles

            if tag != BF_TAG:
                loaded_channels.append(loaded_image)

        loaded_channels = np.array(loaded_channels)
        combined_fluor_img = np.amax(loaded_channels, axis=0)

        circumscribed_combined_fluor_img = drawCirclesOnImage(combined_fluor_img, circles)
        printImage(circumscribed_combined_fluor_img)

    return extracted_circles_dict

def getMaskedDropletObjects(extracted_circles_dict):
    
    max_diameter = max([droplet.shape[0] for droplet in extracted_circles_dict[TAGS[0]]])
    num_droplets = len(extracted_circles_dict[TAGS[0]])
    grid_width = max_diameter+4
    droplet_objects = []
    
    for droplet_id in range(num_droplets):
        droplet_object = {}
        loaded_channels = []
        for tag in extracted_circles_dict:
            channel_img = extracted_circles_dict[tag][droplet_id]
            image_width = channel_img.shape[0]

            sub_segment = np.full((grid_width, grid_width, 3), 0, dtype=np.uint8)
            sx = (grid_width-image_width)/2
            sub_segment[sx:sx+image_width, sx:sx+image_width, :] = channel_img

            a, b = grid_width/2, grid_width/2
            r = channel_img.shape[0]/2
            y,x = np.ogrid[-a:grid_width-a, -b:grid_width-b]
            mask = x*x + y*y > r*r
            sub_segment[mask] = 0
            
            droplet_object[tag+'_blackmask'] = sub_segment.copy()
            
            sub_segment[mask] = 255
            r2 = r + 2
            y,x = np.ogrid[-a:grid_width-a, -b:grid_width-b]
            mask = x*x + y*y > r2*r2
            sub_segment[mask] = 0      
            
            droplet_object[tag+'_whiteline'] = sub_segment.copy()       
        droplet_objects.append(droplet_object)
    return droplet_objects

def printDropletMatrix(droplet_objects, channels, filename=None):
    grid_width = droplet_objects[0][channels[0]].shape[0]
    num_droplets = len(droplet_objects)
    ncolumns = 20
    nrows = num_droplets/ncolumns + 1

    grid_image = np.full((grid_width*nrows, grid_width*ncolumns, 3), 0, dtype=np.uint8)
    
    for droplet_id in range(num_droplets):
        loaded_channels = []
        for tag in channels:
            loaded_channels.append(droplet_objects[droplet_id][tag])
        
        loaded_channels = np.array(loaded_channels)
        combined_img = np.amax(loaded_channels, axis=0)
        
        cy = droplet_id/ncolumns
        cx = droplet_id%ncolumns

        iy = cy*grid_width
        ix = cx*grid_width

        grid_image[iy:iy+grid_width, ix:ix+grid_width, :] = combined_img

    plt.subplots(figsize=(ncolumns, nrows))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid_image)
    if filename:
        plt.imsave(IMAGE_OUTPUT_DIR+filename, grid_image)
    plt.show()

def channelsWithSuffix(channels, suffix):
    return [c+suffix for c in channels]

def dropletMean(droplet, tag):
    return np.mean(droplet[tag+'_blackmask'])

def dropletFluorescenceMean(droplet, tag):
    r = droplet[tag+'_blackmask'].shape[0]
    droplet[tag+'_fluor'] = np.sum(droplet[tag+'_blackmask'])/(np.pi*r*r)
    return droplet

def subsetWhereNonempty(droplets):
    passfilter = []
    failfilter = []
    
    for d in droplets:
        filled = False
        for t in FLUOR_TAGS:
            if dropletMean(d, t) > .5:
                filled = True
        if filled:
            passfilter.append(d)
        else:
            failfilter.append(d)
    return passfilter, failfilter

def countCellsInDrop(droplet, tag, colorMin=(0,0,0), colorMax=(255,255,255), minThresh=0, maxThresh=255, display=False):
    loaded_channel = droplet[tag+'_blackmask']

    mask1 = loaded_channel.copy()
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = minThresh;
    params.maxThreshold = maxThresh;
    params.thresholdStep = 1

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = 5

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask1)

    droplet_keypoints = droplet[tag+'_whiteline'].copy()

    for kp in keypoints:

        kpx = kp.pt[0]
        kpy = kp.pt[1]
        cv2.circle(droplet_keypoints, (int(kpx), int(kpy)), int(kp.size/2), (255, 255, 255), 2)
    
    droplet[tag+'_cellregion'] = droplet_keypoints.copy()
    droplet[tag+'_cellcount'] = len(keypoints)
    
    return droplet

def printSortedDropletMatrix(droplet_objects, channels, filename=None):
    combinations = []
    for r in range(len(channels)+1):
        combinations.extend(itertools.combinations(channels, r))

    def subsetWhereContains(droplets, channel):
        subset = []
        for d in droplets:
            if d[channel+'_cellcount'] > 0:
                subset.append(d)
        return subset

    def subsetWhereNotContains(droplets, channel):
        subset = []
        for d in droplets:
            if d[channel+'_cellcount'] == 0:
                subset.append(d)
        return subset
    
    sortedDroplets = {}
    maxLen = 0
    for in_critera in combinations:

        out_critera = [item for item in channels if item not in in_critera]
        
        subset = droplet_objects
        for in_c in in_critera:
            subset = subsetWhereContains(subset, in_c)
        
        for out_c in out_critera:
            subset = subsetWhereNotContains(subset, out_c)
        
        sortedDroplets[in_critera] = subset
            
        if len(subset) > maxLen:
            maxLen = len(subset)
            
            
    grid_width = droplet_objects[0][channels[0] + '_whiteline'].shape[0]
    num_droplets = len(droplet_objects)
    
    cols_per_segment = 12
    
    ncolumns = len(combinations)*cols_per_segment
    nrows = maxLen/cols_per_segment
    if maxLen%cols_per_segment > 0:
        nrows += 1
        
    grid_image = np.full((grid_width*nrows, grid_width*ncolumns, 3), 0, dtype=np.uint8)
            
    for cx, in_c in enumerate(combinations):
        subset = sortedDroplets[in_c]
        for cy, d in enumerate(subset):
            sc_x = cx * cols_per_segment + cy % cols_per_segment
            sc_y = cy / cols_per_segment
            
            
            ix = sc_x*grid_width
            iy = sc_y*grid_width
            
            loaded_channels = []
            for tag in channels:
                loaded_channels.append(d[tag+'_whiteline'])
            loaded_channels = np.array(loaded_channels)
            combined_img = np.amax(loaded_channels, axis=0) 

            grid_image[iy:iy+grid_width, ix:ix+grid_width, :] = combined_img    

    plt.subplots(figsize=(ncolumns, nrows))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid_image)
    if filename:
        plt.imsave(IMAGE_OUTPUT_DIR+filename, grid_image)
    plt.show()

def dropletsToDF(droplets):    
    validKeys = []
    for key in droplets[0]:
        if type(droplets[0][key]) == float or type(droplets[0][key]) == int:
            validKeys.append(key)
    
    droplets_df = pd.DataFrame(columns=validKeys)
    
    for d in droplets:
        row = {}
        for k in validKeys:
            row[k] = d[k]
        droplets_df = droplets_df.append(row, ignore_index=True)
        
    return droplets_df

def saveInteractionPlot(x, y, z, df, colors, kind, title, filename, size=5, yIsOne=False):
    validKinds = ['bar', 'swarm', 'box']
    if kind not in validKinds:
        print 'invalid kind of plot'
        return

    figsize=(len(FLUOR_TAGS)*3, 8)
    plt.subplots(figsize=figsize)
    
    if kind == 'bar':
        sns.barplot(x=x, y=y, data=df, hue=z, estimator=np.mean, palette=colors)
    
    if kind == 'swarm':
        sns.swarmplot(x=x, y=y, data=df, hue=z, palette=colors, dodge=True, size=size)

    if kind == 'box':
        sns.boxplot(x=x, y=y, data=df, hue=z, palette=colors)        

    if yIsOne:
        x0, x1 = plt.gca().get_xlim()
        plt.plot([x0, x1], [1, 1], color='black', ls='--')

    y0, y1 = plt.gca().get_ylim()
    plt.ylim([0, y1])        
    

    sns.despine()
    plt.title(title)
    plt.legend(title=z, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR+filename, dpi=450, bbox_inches="tight")

def exportInteractionNetwork(pair_df):
    edges_df = pd.DataFrame(columns=['v1', 'v2', 'relative_fitness', 'arrow', 'significant', 'positive', 'edge_scale'])

    output = ''

    for strain in np.unique(pair_df['strain']):
        for partner in np.unique(pair_df['grown with']):
            #absolute fitness with partner
            sub_df = pair_df[pair_df['strain'] == strain]
            sub_df = sub_df[sub_df['grown with'] == partner]
            strain_with_partner_array = sub_df['count']
            absolute_fitness = np.mean(sub_df['count'])


            #fitness by self
            self_df = pair_df[pair_df['grown with'] == strain]
            self_df = self_df[self_df['strain'] == strain]
            strain_alone_array = self_df['count']
            self_fitness = np.mean(self_df['count'])


            relative_fitness = absolute_fitness/self_fitness
            arrow = relative_fitness > 1

            p_value = sp.stats.ttest_ind(strain_alone_array, strain_with_partner_array, equal_var = False)[1]

            std = np.std(strain_with_partner_array/self_fitness)

            output += '\t'.join([partner, 'to', strain,
                'fitness:', str(np.around(relative_fitness, decimals=2)),
                'std:', str(std),
                'p value:', str(p_value)]) + '\n'


            significant = p_value < 0.05
            
            edge_scale = np.around(np.log2(relative_fitness), decimals=2)

            edges_df = edges_df.append({'v1':partner, 'v2':strain, 'relative_fitness':np.around(relative_fitness, decimals=2),
                                        'arrow':arrow, 'significant':significant, 'positive':relative_fitness>1,
                                       'edge_scale': edge_scale},
                                      ignore_index=True)

    print output
    open(PLOT_OUTPUT_DIR+'p_values.txt', 'w').write(output)
    edges_df.to_csv(PLOT_OUTPUT_DIR+'cytoscape_edges.csv')