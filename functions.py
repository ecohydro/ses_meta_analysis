from collections import defaultdict
import pandas as pd
import numpy as np
import plotly.graph_objs as go
PI = np.pi

def create_matrix(dataframe, code=0):
    """ Generates the analysis matrix for our SES metadata CIRCOS figure.
    
        Input:
            dataframe - this is the datafame generated from
                the Circos_DATA.csv file.
        
            code - A code (1-3) that specifies which subset of the 
                manuscripts to analyze. Default is zero.
                
                    0 = Analyze all manuscripts
                    1 = Analyze manuscripts that are coded as Social
                        Impacts on Environmental Systems
                    2 = Analyze manuscripts that are coded as Environmental
                        Impacts on Social Systems
                    3 = Analyze manuscripts that are coded as Coupled
                        Systems Dynamics
    
    """
    if code == 0:
        df = dataframe.loc[dataframe['Code'] > 0]
    else:
        # Filter the data based on the value of the code argument.
        df = dataframe.loc[dataframe['Code'] == code]
    
    # Initialize the circos_matrix as a defaultdict so that we do not
    # need to specify the dict keys before assignment.
    circos_matrix= defaultdict( lambda:
        defaultdict(lambda: defaultdict(int)))

    # Get the list of datatypes using the column headers of the dataframe.
    #
    # WARNING: This assumes that the dataframe has the structure defined in the
    #   ipython notebook, `Circos Diagram.ipynb`. Any changes will break
    #   this code.
    #
    datatypes = list(df.columns[2:-1])

    for datatype in datatypes:
        # 0. Assign the datatype-datatype node to zero:
        circos_matrix[datatype][datatype] = 0
        # 1. Assign all types to othertypes
        # NOTE: We can't use same list as in for loop!
        othertypes = list(df.columns[2:-1])
        # 2. Remove the current datatype from othertypes
        othertypes.remove(datatype)
        # 3. Iterate over all the remaining othertypes:
        for othertype in othertypes:
            # 4. Initialize this combination to zero (may be default)
            circos_matrix[datatype][othertype] = 0
            # 5. Find all papers containing this combination of types
            matches = len(df.loc[(df[datatype] == 1) & (df[othertype] == 1)])
            # 6. Assign the # of matches to the current combination of types
            circos_matrix[datatype][othertype] = matches

    # Return our result as a pandas dataframe instead of a dict.
    return pd.DataFrame(circos_matrix)



def moduloAB(x, a, b):
    """ Maps a real number onto the unit circle
    identified with the interval [a,b), b-a=2*PI.
    """

    if a>=b:
        raise ValueError('Incorret interval ends')

    y = (x-a)%(b-a)
    return y+b if y<0 else y+a


def test_2PI(x):
    return 0<= x < 2*np.pi


def check_data(data_matrix):
    L, M=data_matrix.shape
    if L!=M:
        raise ValueError('Data array must have (n,n) shape')
    return L


def get_ideogram_ends(ideogram_len, gap):
    ideo_ends=[]
    left=0
    for k in range(len(ideogram_len)):
        right=left+ideogram_len[k]
        ideo_ends.append([left, right])
        left=right+gap
    return ideo_ends


def make_ideogram_arc(R, phi, a=50):
    # R is the circle radius
    # phi is the list of ends angle coordinates of an arc
    # a is a parameter that controls the number of points to be evaluated on an arc
    if not test_2PI(phi[0]) or not test_2PI(phi[1]):
        phi=[moduloAB(t, 0, 2*PI) for t in phi]
    length=(phi[1]-phi[0])% 2*PI
    nr=5 if length<=PI/4 else int(a*length/PI)

    if phi[0] < phi[1]:
        theta=np.linspace(phi[0], phi[1], nr)
    else:
        phi=[moduloAB(t, -PI, PI) for t in phi]
        theta=np.linspace(phi[0], phi[1], nr)
    return R*np.exp(1j*theta)


def map_data(data_matrix, row_value, ideogram_length):
    L = check_data(data_matrix)
    mapped=np.zeros(data_matrix.shape)
    for j  in range(L):
        mapped[:, j]=ideogram_length*data_matrix[:,j]/row_value
    return mapped


def make_ribbon_ends(mapped_data, ideo_ends,  idx_sort):
    L=mapped_data.shape[0]
    ribbon_boundary=np.zeros((L,L+1))
    for k in range(L):
        start=ideo_ends[k][0]
        ribbon_boundary[k][0]=start
        for j in range(1,L+1):
            J=idx_sort[k][j-1]
            ribbon_boundary[k][j]=start+mapped_data[k][J]
            start=ribbon_boundary[k][j]
    return [[(ribbon_boundary[k][j],ribbon_boundary[k][j+1] ) for j in range(L)] for k in range(L)]


def control_pts(angle, radius):
    #angle is a  3-list containing angular coordinates of the control points b0, b1, b2
    #radius is the distance from b1 to the  origin O(0,0) 

    if len(angle)!=3:
        raise InvalidInputError('angle must have len =3')
    b_cplx=np.array([np.exp(1j*angle[k]) for k in range(3)])
    b_cplx[1]=radius*b_cplx[1]
    return zip(b_cplx.real, b_cplx.imag)


def ctrl_rib_chords(l, r, radius):
    # this function returns a 2-list containing control poligons of the two quadratic Bezier
    #curves that are opposite sides in a ribbon
    #l (r) the list of angular variables of the ribbon arc ends defining 
    #the ribbon starting (ending) arc 
    # radius is a common parameter for both control polygons
    if len(l)!=2 or len(r)!=2:
        raise ValueError('the arc ends must be elements in a list of len 2')
    return [control_pts([l[j], (l[j]+r[j])/2, r[j]], radius) for j in range(2)]


def make_q_bezier(b):# defines the Plotly SVG path for a quadratic Bezier curve defined by the 
                     #list of its control points
    test_data = list(b)
    if len(test_data)!=3:
        raise ValueError('control poligon must have 3 points')
    A, B, C=b
    return 'M '+str(A[0])+',' +str(A[1])+' '+'Q '+\
                str(B[0])+', '+str(B[1])+ ' '+\
                str(C[0])+', '+str(C[1])


def make_ribbon_arc(theta0, theta1):

    if test_2PI(theta0) and test_2PI(theta1):
        if theta0 < theta1:
            theta0= moduloAB(theta0, -PI, PI)
            theta1= moduloAB(theta1, -PI, PI)
            if theta0*theta1>0:
                raise ValueError('incorrect angle coordinates for ribbon')

        nr=int(40*(theta0-theta1)/PI)
        if nr<=2: nr=3
        theta=np.linspace(theta0, theta1, nr)
        pts=np.exp(1j*theta)# points on arc in polar complex form

        string_arc=''
        for k in range(len(theta)):
            string_arc+='L '+str(pts.real[k])+', '+str(pts.imag[k])+' '
        return   string_arc
    else:
        raise ValueError('the angle coordinates for an arc side of a ribbon must be in [0, 2*pi]')


def make_layout(title, plot_size):
    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

    return go.Layout(title=title,
                  xaxis=dict(axis),
                  yaxis=dict(axis),
                  showlegend=False,
                  width=plot_size,
                  height=plot_size,
                  margin=dict(t=25, b=25, l=25, r=25),
                  hovermode='closest',
                  shapes=[]# to this list one appends below the dicts defining the ribbon,
                           #respectively the ideogram shapes
                 )

def make_ideo_shape(path, line_color, fill_color):
    #line_color is the color of the shape boundary
    #fill_collor is the color assigned to an ideogram
    return  dict(
                  line=dict(
                  color=line_color,
                  width=0.45
                 ),

            path=  path,
            type='path',
            fillcolor=fill_color,
            layer='below'
        )


def make_ribbon(l, r, line_color, fill_color, radius=0.2):
    #l=[l[0], l[1]], r=[r[0], r[1]]  represent the opposite arcs in the ribbon 
    #line_color is the color of the shape boundary
    #fill_color is the fill color for the ribbon shape
    poligon=ctrl_rib_chords(l,r, radius)
    # Need to use map to coerce the zip iterables into lists.
    b,c = list(map(list, poligon))

    return  dict(
                line=dict(
                color=line_color, width=0.5
            ),
            path=  make_q_bezier(b)+make_ribbon_arc(r[0], r[1])+
                   make_q_bezier(c[::-1])+make_ribbon_arc(l[1], l[0]),
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

def make_self_rel(l, line_color, fill_color, radius):
    #radius is the radius of Bezier control point b_1
    b=control_pts([l[0], (l[0]+l[1])/2, l[1]], radius)
    return  dict(
                line=dict(
                color=line_color, width=0.5
            ),
            path=  make_q_bezier(b)+make_ribbon_arc(l[1], l[0]),
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

def invPerm(perm):
    # function that returns the inverse of a permutation, perm
    inv = [0] * len(perm)
    for i, s in enumerate(perm):
        inv[s] = i
    return inv

