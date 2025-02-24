import sys, pdb, argparse, ovito, time, random, math
import numpy as np
import quaternion
import fil_io as io
from ovito.io import *
from ovito.modifiers import *
from scipy.spatial.distance import pdist, squareform

class progressbar:
  def __init__(self,minlen=10,symbol='-'):
    import fcntl, termios, struct, os
    self.fd = os.open(os.ctermid(), os.O_RDONLY)
    (self.height,self.width) = struct.unpack('hh', fcntl.ioctl(self.fd, termios.TIOCGWINSZ, '1234'))
    self.symbol = symbol
    self.minlen = minlen

  def update(self,percentage,message):
    import fcntl, termios, struct
    (self.height,self.width) = struct.unpack('hh', fcntl.ioctl(self.fd, termios.TIOCGWINSZ, '1234'))
    message = message.ljust(self.minlen)                    # fill message to minlen with spaces
    meslen = len(message)
    barsize = max(0,int((self.width-meslen)*percentage)-2)  # progress bar size : -2 for [,]
    filsize = self.width - meslen - barsize - 2             # fill with spaces to end of line
    sys.stderr.write('\r'+message+'[')
    for i in range(barsize): sys.stderr.write(self.symbol[i%len(self.symbol)])
    for i in range(filsize): sys.stderr.write(' ')
    sys.stderr.write(']')
    sys.stderr.flush()

  def updateinitial(self,message,symbol=None):
    if symbol: self.symbol = symbol
    self.update(0,message)

  def updatefinal(self,message):
    self.update(1,message)
    sys.stderr.write('\n')

class group:
    def __init__(self, group, origin, bonded_pos, box, aux_vec= np.zeros(3), surface=0):
        self.origin = origin
        if surface > 0:
            self.surface = surface
        else:
            if origin[-1] > box[-1]*0.5:
                self.surface = 1
            else:
                self.surface = -1

        if group == 'hydroxyl':
            self.labels = ['CFO1','CFH1']
            self.natoms = 2
            self.group = group

            self.positions = np.array([[0.,0.,1.4],[0.,0.,2.3]]) * self.surface + origin
            self.aux_vec = aux_vec

        elif group == 'epoxy':
            self.labels = ['CFO2',]
            self.natoms = 1
            self.group = group
            self.positions = np.array([[0.,0.,1.2],]) * self.surface + origin
            self.aux_vec = aux_vec

        elif group == 'H':
            self.labels = ['CFH2',]
            self.natoms = 1
            self.group = group
            align_vec = (bonded_pos[1] - bonded_pos[0])*0.5 + bonded_pos[0]
            align_vec = origin - align_vec
            align_vec = 1.1 * align_vec / np.linalg.norm(align_vec)
            self.positions = np.array([align_vec + origin, ])
            self.aux_vec = aux_vec

        elif group == 'H2':
            self.labels = ['CFH3', 'CFH3']
            self.natoms = 2
            self.group = group
            align_vec = (bonded_pos[1] - bonded_pos[0])*0.5 + bonded_pos[0]
            align_vec = origin - align_vec
            align_vec = 1.1 * align_vec / np.linalg.norm(align_vec)
            p1 = rotate_about_vec(align_vec, bonded_pos[1] - bonded_pos[0], np.pi* 50./180.)
            p2 = rotate_about_vec(align_vec, bonded_pos[1] - bonded_pos[0], -np.pi* 50./180.)
            self.positions = np.array([p1, p2])
            self.aux_vec = aux_vec

        elif group == 'carbonyl':
            self.labels = ['CFO3',]
            self.natoms = 1
            self.group = group
            align_vec = (bonded_pos[1] - bonded_pos[0])*0.5 + bonded_pos[0]
            align_vec = origin - align_vec
            align_vec = 1.2 * align_vec / np.linalg.norm(align_vec)
            self.positions = np.array([align_vec + origin, ])
            self.aux_vec = aux_vec

        elif group == 'COOH':
            self.labels = ['CFO4', 'CFO5', 'CF9', 'CFH4']
            # O O C H coords
            els= np.array([[-2.6494655148,      0.8764317572,      0.1513536183],
                           [-0.8355786901,      1.0157882276,     -0.3418991825],
                           [-1.7121316364,      0.1579877798,     -0.0282781848],
                           [-0.7955422515,      1.9195389217,     -0.4432936313]])
            els[:] -= els[2,:]
            els = els[:,[2,0,1]] * self.surface + origin + np.array([0,0,1.45]) * self.surface
            self.natoms = 3
            self.group = group
            self.positions = els
            self.aux_vec = aux_vec

    def check_and_move(self, near_positions, box, r_cut, pbc=[True, True, False], count_limit= 10000):
        is_good = False
        flip_check = False
        tmp = []
        counts = 0
        # initial check to see if anything is too close
        for p in self.positions:
            dists = distance(p, near_positions, box, pbc)
            #dists = np.zeros(near_positions.shape[0])
            #functions.distances(p, near_positions, box, pbc, dists)
            tmp.append(dists < r_cut)
        num_close = np.array(tmp).flatten().sum()
        if num_close > 0:
            while counts < count_limit:
                if self.group == 'epoxy':
                    v = self.aux_vec
                    v -= self.origin
                else:
                    v = np.array([random.random() for _ in range(3)])
                for i,p in enumerate(self.positions):
                    self.positions[i] = rotate_about_vec(p - self.origin, v, np.pi/15.) + self.origin
                is_close = []

                for p in self.positions:
                    dists = distance(p, near_positions, box, pbc)
                    #dists = np.zeros(near_positions.shape[0])
                    #functions.distances(p, near_positions, box, pbc, dists)
                    is_close.append(np.any(dists < r_cut))


                num_close = np.array(is_close).flatten().sum()
                flip_check = self.check_flip_status()
                counts += 1
                if (num_close == 0) and (flip_check):
                    is_good = True
                    break
        else:
            is_good = True
        return is_good

    def check_flip_status(self):
        z = self.positions[0][-1] - self.origin[-1]
        if  self.surface == 1:
            if z > 0.0:
                flip_status = True
            else:
                flip_status = False
        elif self.surface == -1:
            if z < 0.:
                flip_status = True
            else:
                flip_status = False

        return flip_status

def find_surface_atoms(infile, box, pbc, surf_rad):
    pipeline = import_file(infile, columns = ["Particle Type", "Position.X", "Position.Y", "Position.Z"])
    ## Setup the pipeline
    unit_cell = np.zeros((3,4))
    for i in range(3):
        unit_cell[i,i] = box[i]

    #data.cell_.pbc = tuple(pbc)
    data = pipeline.compute()
    data.cell_[:,:] = unit_cell
    data.apply(AffineTransformationModifier(operate_on= {'cell'},
                                            relative_mode= False,
                                            target_cell= unit_cell))
    pipeline.modifiers.append(ConstructSurfaceModifier(radius = surf_rad,
                                                       only_selected = False,
                                                       select_surface_particles= True))
    data = pipeline.compute()
    sel = np.array(data.particles_.selection_).astype(int)
    return sel

def cart2sphere(x):
    # x,y,z -> r,theta,phi
    r = np.linalg.norm(x)
    theta = np.arccos(x[2]/r)
    phi = np.arctan2(x[1]/x[0])
    return np.array([r,theta,phi])

def sphere2cart(b):
    # r,theta,phi -> x,y,z
    x = b[0] * np.sin(b[1]) * np.cos(b[2])
    y = b[0] * np.sin(b[1]) * np.sin(b[2])
    z = b[0] * np.cos(b[1])
    return np.array([x,y,z])

def distance(x0, x1, box, pbc=[True, True, True]):
    # xo is a position of one atom, x1 is an array of positions
    # use the pbc bool mask to set the periodicity
    delta = np.abs(x0 - x1)
    delta[:,pbc] -= box[pbc] * np.round(delta[:,pbc]/(box[pbc]))
    return np.sqrt((delta ** 2).sum(axis=-1))

def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j

def distance_fast(pos, box, pbc=[True, True, True]):
    dist_nd_sq = np.zeros(pos.shape[0] * (pos.shape[0] - 1) // 2)
    for dim in range(pos.shape[1]):
        pos_1d = pos[:, dim][:, np.newaxis]
        dist_1d = pdist(pos_1d)
        #if pbc[dim]: dist_1d[dist_1d > box[dim]*0.5] -= box[dim]
        if pbc[dim]: dist_1d -= box[dim] * np.round(dist_1d/box[dim])
        dist_nd_sq += dist_1d ** 2
    # returns the condensed distance matrix
    # dist(i,j) == natom * i + j - ((i+2) * (i+1))//2
    return np.sqrt(dist_nd_sq)

def distance_even_faster(pos, box, pbc=[True, True, True]):
    n = pos.shape[0]  # Number of atoms
    dist_nd_sq = np.zeros(n * (n - 1) // 2)  # Condensed distance matrix
    k = 0
    for i in range(n - 1):
        # Pairwise differences for all remaining atoms
        delta = pos[i + 1:] - pos[i]

        # Apply PBC to the necessary dimensions
        for dim in range(pos.shape[1]):
            if pbc[dim]:
                delta[:, dim] -= box[dim] * np.round(delta[:, dim] / box[dim])

        # Compute squared distances
        dist_nd_sq[k:k + n - i - 1] = np.sum(delta ** 2, axis=1)
        k += n - i - 1

    # Return actual distances by taking the square root
    return np.sqrt(dist_nd_sq)

def condmat2coord(nd_sq, natoms, cutoff=1.95):
    coord = np.zeros(natoms)
    mask = nd_sq < cutoff
    inds = np.arange(nd_sq.shape[0])
    bonds = [condensed_to_square(ii, natoms) for ii in inds[mask]]
    for pair in bonds:
        coord[pair[0]] += 1
        coord[pair[1]] += 1

    # This works but is crazy slow! vectorise it!!!
    #=============================================
    #for j in range(natoms):
    #    for i in range(j):
    #        ind = natoms * i + j - ((i+2) * (i+1))//2
    #        if (nd_sq[ind]<1.95):
    #            coord[i] += 1
    #            coord[j] += 1


    return coord

def rotate_about_vec(u,a,theta):
    # rotate vector u, by an angle theta, about an arbitrary vector a, using quaternions.
    a = a / np.linalg.norm(a)
    q1 = np.quaternion(0.,u[0],u[1],u[2])
    q2 = np.quaternion(np.cos(theta/2.), a[0] * np.sin(theta/2.), a[1] * np.sin(theta/2.), a[2]*np.sin(theta/2.))
    q3 = q2 * q1 * np.conjugate(q2)
    return quaternion.as_float_array(q3)[1:]

if __name__=="__main__":
    # main code
    parser = argparse.ArgumentParser(description='Program install oxygen groups (hydroxyl and epoxy) onto pristine graphene surface.')

    #================#
    # ordered inputs #
    #================#
    parser.add_argument('infile', metavar='infile', type=str,
                        help='input xyz file, can be in extended form with proper header. Particle type C1 must denote exposed surface carbons, set all other carbon types to C2.')

    #================#
    # optional flags #
    #================#
    parser.add_argument('--cutoff',dest='r_cut',default=1.95, type=float,
                        help='Change value of the bond distance cutoff that is used for ALL particles. Default is 1.95 Å.')
    parser.add_argument('--random_seed',dest='seed',default=666, type=int,
                        help='Change the random number seed, default is 666')
    parser.add_argument('--cov_fact',dest='cov_fact',default=0.4, type=float,
                        help='Coverage factor of oxygen groups on surface, float between 0.0 -> 1.0')
    parser.add_argument('--sep',dest='sep',default=1.851, type=float,
                        help='minimum separation distance between inserted particles and all others, default is 1.851 Å')
    parser.add_argument('--count_limit',dest='count_limit',default=10000, type=int,
                        help='Change the random number seed, default is 666')
    parser.add_argument('--silent', dest='silent', default=False, action='store_true',
                       help="Run program silently")
    parser.add_argument('--hydroxyl_epoxy_ratio', default=0.7, type=list,
                       help="probability of hydroxyl group installation out of 1, all other instances will install epoxy groups")
    parser.add_argument('--pbc', default=[True, True, True], type=list,
                        help="periodic boundary conditions, by default there no periodicity in z, i.e. is set to: [True, True, False]")
    parser.add_argument('--check_distances', dest='check_distances', default=False, action='store_true',
                       help="Double check inter-atomic distances after additional atoms have been installed. Default is False.")
    parser.add_argument('--outfile', dest='outfile', default=None, type=str,
                       help="Specify name of output data file, default is the inputfilename with 'MOD.xyz' appended instead of the existing extension.")
    parser.add_argument('--guess_rcut', default=False, action='store_true',
                       help="Guess r_cut from particle labels. Default is False.")
    parser.add_argument('--surface_probe_radius', dest='surf_rad', default=4.0, type=float,
                       help="Specify the surface probe radius for surface mesh calculation. Default is 4.0 Å.")

    args = parser.parse_args()

    # assign inputs to some normal vars
    infile = args.infile
    r_cut = args.r_cut
    seed = args.seed
    count_limit = args.count_limit
    cov_fact = args.cov_fact
    oh2o_ratio = args.hydroxyl_epoxy_ratio
    silent = args.silent
    sep = args.sep
    pbc = args.pbc
    check_dists = args.check_distances
    outfile = args.outfile
    guess_rcut = args.guess_rcut
    surf_rad = args.surf_rad

    if not silent:
        print(f'Coverage Factor: {cov_fact}')

    # seed is default 666
    random.seed(seed)

    if not outfile: outfile = infile.split('.')[0] + '_MOD.xyz'
    if not silent: print(f'Reading input from file {infile}\n')

    # get data from xyz file
    natoms, box, aids, atypes, atypelabels, apos, unique_atypes, unique_atypelabels = io.read_xyz(infile)
    print(atypes)
    print(atypelabels)

    # set cell origin so that we can use the easy cell boundary condition
    box = np.copy(box)
    apos[:,pbc] -= box[pbc] * 0.5
    apos[:,pbc] -= box[pbc] * np.round(apos[:,pbc]/(box[pbc]))

    # find atoms that form the surface, via aovito subroutine
    if not silent: print('Finding surface atoms')
    is_surface = find_surface_atoms(infile, box, pbc, surf_rad)
    #box[-1] = apos[:,-1].max() - apos[:,-1].min() + 2.

    io.write_xyz(natoms, atypelabels, apos, box, 'tmp.xzy')

    # determine coordination and give atoms a type label
    #p = progressbar(25)
    #p.updateinitial('Determining coordination','-+')

    acoords = np.zeros(natoms)

    # Old distance calculation routine
    #t0 = time.time()
    #for i, i_pos in enumerate(apos):
    #    percentage = float(i)/float(natoms-1)
    #    p.update(percentage,'Determining atom coordination')
    #    dists = distance(i_pos, apos, box)
    #    mask = np.logical_and( dists > 0.0001, dists <= 1.95 )
    #    acoords[i] = mask.sum()

    #t1 = time.time()
    #print(f'old distance routine: {t1-t0} s)')

    t0 = time.time()
    #dist_mat = distance_fast(apos, box)
    dist_mat = distance_even_faster(apos, box)
    acoords2 = condmat2coord(dist_mat,natoms)
    acoords = acoords2
    t1 = time.time()
    print(f'new distance routine: {t1-t0} s')

    io.write_xyz(natoms, atypelabels, apos, box, 'tmp.xzy', extra=[acoords, acoords2])

    for coord in np.unique(acoords):
        mask = acoords == coord
        if coord == 2:
            atypelabels[mask] = 'CF2'
        elif coord == 3:
            atypelabels[mask] = 'CF1'
        elif coord == 4:
            atypelabels[mask] = 'CF4'

    unique_atypelabels = list(np.unique(atypelabels))

    if not silent:
        for t in unique_atypelabels:
            print(f'{(atypelabels==t).sum()} ({np.round((atypelabels==t).sum()/float(natoms),1)}%) {t} atoms')

    if not silent: print(f'{is_surface.sum()} surface carbons\n')
    #apos[:,-1] -= apos[:,-1].min() + 1.
    natomtypes = len(unique_atypes)
    print(natoms, box)
    p = progressbar(25)
    p.updateinitial('installing edge groups','-+')
    added_groups = []
    denied_hydroxyl = 0
    denied_epoxy = 0
    denied_carboxyl = 0
    denied_carbonyl = 0

    #r_cut = np.full(natoms, rcut)
    #if guess_rcut:
    #    for lab in unique_atypelabels:
    #        mask = atypelabels == lab
    #        no_num_lab = ''.join([i for i in lab if not in i.isdigit()])
    #        if no_num_lab == 'CF':
    #            r_cut[mask] = 1.85
    #        elif no_num_lab == 'CFO':
    #            r_cut[mask] = 1.85
    #        elif no_num_lab == 'CFH':
    #            r_cut[mask] = 1.2
    #        elif no_num_lab == 'CFN':
    #            r_cut[mask] = 1.85

    # first saturate the all the edges
    # point edges first
    num_point = (atypelabels == 'CF3').sum()
    num_edge = (atypelabels == 'CF2').sum()
    num_sites = num_point + num_edge
    mask = atypelabels == 'CF3'
    # convert point edges to ethers
    atypelabels[mask] = 'CFO6'
    num_ether = num_point
    mask = atypelabels == 'CFO6'
    idx = np.where(mask)[0]
    for i,(c_id, c_pos, c_idx) in enumerate(zip(aids[mask], apos[mask], idx)):
        dists = distance(c_pos, apos, box, pbc= pbc)
        bonded_mask = np.logical_and(dists < r_cut, dists > 0.01)
        for b_id in aids[bonded_mask]:
            atypelabels[aids == b_id] = 'CF10'

    # now edges
    mask = atypelabels == 'CF2'
    idx = np.where(mask)[0]
    for i,(c_id, c_pos, c_label, c_idx) in enumerate(zip(aids[mask], apos[mask], atypelabels[mask], idx)):
        percentage = float(i)/float(num_edge)
        p.update(percentage,'installing edge groups (%d/%d)'%(i+1,num_edge))
        if len(added_groups)>0:
            n = len(added_groups)
            all_pos = np.concatenate((apos, np.vstack([g.positions for g in added_groups])))
        else:
            all_pos = apos

        # find neighbour carbons bonded to target carbon
        dists = distance(c_pos, apos, box, pbc= pbc)
        #dists = np.zeros(apos.shape[0])
        #functions.distances(c_pos, apos, box, pbc, dists)
        bonded_mask = np.logical_and(dists < r_cut, dists > 0.01)
        bonded_pos = apos[bonded_mask]
        bonded_labels = atypelabels[bonded_mask]
        bonded_ids = aids[bonded_mask]

        if len(bonded_ids) < 2:
            pdb.set_trace()

        # find ALL atoms withn 12A of the target carbon
        dists = distance(c_pos, all_pos, box, pbc=pbc)
        #dists = np.zeros(all_pos.shape[0])
        #functions.distances(c_pos, all_pos, box, pbc, dists)
        all_dist_mask = (dists < 12.) & (dists > 0.01)
        near_pos = all_pos[all_dist_mask].copy()
        near_pos[:,pbc] -= box[pbc]*np.round(near_pos[:,pbc]/box[pbc])

        # roll the dice to decide which group is to be installed
        rand = random.random()
        # carboxylic acid
        if rand < 0.6:
            g = group('COOH', c_pos, bonded_pos, box)
            status = g.check_and_move(near_pos, box, sep, count_limit= count_limit)
            if status == True:
                added_groups.append(g)
                atypelabels[c_idx] = 'CF7'
            else:
                denied_carboxyl += 1
                status = False
                # force fit a hydrogen if nothing fits
                while not status:
                    g = group('H', c_pos, bonded_pos, box)
                    # see if manually reducing sep to 1.2 helps
                    status = g.check_and_move(near_pos, box, 1.2, count_limit= count_limit)
                    if status == True:
                        added_groups.append(g)
                        atypelabels[c_idx] = 'CF2'
                        break

        else:
            g = group('carbonyl', c_pos, bonded_pos, box)
            status = g.check_and_move(near_pos, box, sep, count_limit= count_limit)
            if status == True:
                added_groups.append(g)
                atypelabels[c_idx] = 'CF8'
            else:
                # force fit a hydrogen if nothing fits
                denied_carbonyl += 1
                status = False
                while not status:
                    g = group('H', c_pos, bonded_pos, box)
                    # see if manually reducing sep to 1.2 helps
                    status = g.check_and_move(near_pos, box, 1.2, count_limit= count_limit)
                    if status == True:
                        added_groups.append(g)
                        atypelabels[c_idx] = 'CF2'
                        break

        if i+1==num_sites: p.updatefinal('install edge groups (%d/%d)'%(num_edge,num_edge))

    if len(added_groups) > 0:
        tmp_pos = np.vstack((apos, np.vstack([g.positions for g in added_groups])))
        tmp_labels = np.concatenate((atypelabels, np.array([el for g in added_groups for el in g.labels], dtype=str)))

    io.write_xyz(len(tmp_pos), tmp_labels, tmp_pos, box, infile.split('.')[0] + '_edge_only.xyz')

    # select C1 surface atoms that are determined to be on the surface for functionalisation
    surface_atype = 'CF1'
    surface_mask = (atypelabels == surface_atype) & (is_surface)
    chance = np.array([random.random() for _ in range(natoms)])
    mask = np.logical_and( surface_mask, chance <= cov_fact)
    idx = np.where(mask)[0]

    # iterate over selected surface target carbons
    for i,(c_id, c_pos, c_label, c_idx) in enumerate(zip(aids[mask], apos[mask], atypelabels[mask], idx)):
        num_sites = mask.sum()
        # update the progress bar
        percentage = float(i)/float(num_sites)
        p.update(percentage,'installing surface groups (%d/%d)'%(i+1,num_sites))
        # add the new atom positions so they can be avoided when further adding groups groups
        if len(added_groups)>0:
            n = len(added_groups)
            all_pos = np.concatenate((apos, np.vstack([g.positions for g in added_groups])))
        else:
            all_pos = apos

        # find thickness of substrate in local domain about the target
        dists = distance(c_pos[:-1], apos[:,:-1], box[:-1], pbc=[True, True])
        local_mask = np.logical_and( dists < 5., dists > 0.01)
        local_apos = apos[local_mask]
        local_height = local_apos[:,-1].max() - local_apos[:,-1].min()
        tmp = c_pos[-1] - local_apos[:,-1].min()
        if tmp > (0.5 * local_height):
            surface = 1
        else:
            surface = -1

        # find neighbour carbons bonded to target carbon
        dists = distance(c_pos, apos, box, pbc= pbc)
        #dists = np.zeros(apos.shape[0])
        #functions.distances(c_pos, apos, box, pbc, dists)
        bonded_mask = np.logical_and(dists < r_cut, dists > 0.01)
        bonded_pos = apos[bonded_mask]
        bonded_labels = atypelabels[bonded_mask]
        bonded_ids = aids[bonded_mask]

        # find ALL atoms withn 4A of the target carbon
        dists = distance(c_pos, all_pos, box, pbc=pbc)
        #dists = np.zeros(all_pos.shape[0])
        #functions.distances(c_pos, all_pos, box, pbc, dists)
        all_dist_mask = (dists < 12.) & (dists > 0.01)
        near_pos = all_pos[all_dist_mask].copy()

        # roll the dice to decide which group is to be installed
        rand = random.random()
        # hydroxyl
        if (rand < oh2o_ratio) and (atypelabels[aids==c_id]=='CF1'):
            g = group('hydroxyl', c_pos, bonded_pos, box, surface= surface)
            status = g.check_and_move(near_pos, box, sep, count_limit= count_limit)
            if status == True:
                added_groups.append(g)
                atypelabels[c_idx] = 'CF5'
            else:
                denied_hydroxyl += 1

        # epoxy
        elif rand > oh2o_ratio:
            #try to attach epoxy group iterating over the bonded neighbours of the target carbon, stop if we succeed
            for b_idx in range(3):
                # check that both site carbons of the potential epoxy site are not adjacent to another epoxy carbon
                dists = distance(bonded_pos[b_idx], apos, box, pbc= pbc)
                #dists = np.zeros(apos.shape[0])
                #functions.distances(bonded_pos[b_idx], apos, box, pbc, dists)
                nextto_bonded = (dists < r_cut) & (dists > 0.01)
                nextto_labels = atypelabels[nextto_bonded]
                tmp = np.concatenate((nextto_labels,bonded_labels))
                if ('CF6' not in tmp) and ('CF2' not in tmp):
                    # check that there is nothing bonded to the other site carbon
                    if (bonded_labels[b_idx] == 'CF1'):
                        # get vector between the two carbon bonding sites
                        tmp = bonded_pos[b_idx] - c_pos
                        tmp[pbc] -= box[pbc]  * np.round(tmp[pbc]/box[pbc])
                        origin = tmp * 0.499 + c_pos
                        # the origin of the epoxy is the midpoint of this vector
                        origin[pbc] -= box[pbc] * np.round(origin[pbc]/box[pbc])
                        g = group('epoxy', origin, bonded_pos, box, aux_vec= tmp+c_pos, surface= surface)
                        # for epoxy, remove the other bonded carbon from the array of near positions
                        status = g.check_and_move(near_pos[[not np.array_equal(bonded_pos[b_idx], pp) for pp in near_pos]],
                                                  box,
                                                  sep,
                                                  count_limit= count_limit)
                        if status == True:
                            added_groups.append(g)
                            atypelabels[bonded_ids[b_idx] == aids] = 'CF6'
                            atypelabels[c_idx] = 'CF6'
                            break
            if status == False:
                denied_epoxy += 1
        if i+1==num_sites: p.updatefinal('install groups (%d/%d)'%(num_sites,num_sites))


    if len(added_groups) > 0:
        apos = np.vstack((apos, np.vstack([g.positions for g in added_groups])))
        atypelabels = np.concatenate((atypelabels, np.array([el for g in added_groups for el in g.labels], dtype=str)))

    for i, ipos in enumerate(apos[:-1]):
        dists = distance(ipos, apos[i+1:], box, pbc= pbc)
        #dists= np.zeros(apos[i+1:].shape[0])
        #functions.distances(ipos, apos[i+1:], box, pbc, dists)
        if np.any(dists < sep):
            j = np.where(dists<0.8)[0] + i+1
            for jj in j:
                print('Problem!!')
                print(f'atoms {i} and {jj} are too close, {atypelabels[i]}-{atypelabels[jj]}')

    newbox = box.copy()
    #natoms += int(np.array([g.natoms for g in added_groups]).sum())
    natoms = apos.shape[0]
    #apos[:,pbc] -= box[pbc] * np.round(apos[:,pbc]/box[pbc])
    apos[:,pbc] += box[pbc] * 0.5
    #apos[:,-1] += 20.0
    #min_pos = np.min(apos[:,-1]) - 4
    #apos[:,-1] -= min_pos
    #newbox[2] = np.max(apos[:,2]) + 40.0

    print(f'{denied_hydroxyl} hydroxyls and {denied_epoxy} epoxies failed to install')

    print('Writing result to file')
    io.write_xyz(natoms, atypelabels, apos, newbox, outfile)

    print('Writing log file')
    with open('log.txt','w') as fo:
        fo.write(f'Coverage factor: {cov_fact}\n')
        fo.write(f'Hydroxyl probability: {oh2o_ratio}, epoxy: {1. - oh2o_ratio}\n')
        fo.write(f'Fitting attempts per group: {count_limit}\n')
        fo.write(f'{num_point} ethers\n')
        types, counts = np.unique(np.array([g.group for g in added_groups]), return_counts= True)
        for tt, num in zip (types, counts):
            fo.write(f'{num} {tt}\n')
        for i,g in enumerate(added_groups):
            fo.write(f'{i+1} {g.group} {g.surface}\n')
