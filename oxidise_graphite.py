import numpy as np
import functions
import random
import fil_io as io
import argparse
import pdb
import sys
import ovito
from ovito.io import *
from ovito.modifiers import *

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
    def __init__(self, group, origin, box):
        self.origin = origin
        if group == 'hydroxyl':
            self.labels = ['O1','H1']
            if origin[-1] > box[-1]*0.5:
                self.surface = 1
            else:
                self.surface = -1
            self.natoms = 2
            self.group = group
            self.positions = np.array([[0.,0.,1.4],[0.,0.,2.3]]) * self.surface + origin

        elif group == 'epoxy':
            self.labels = ['O2',]
            if origin[-1] > box[-1]*0.5:
                self.surface = 1
            else:
                self.surface = -1
            self.natoms = 1
            self.group = group
            self.positions = np.array([[0.,0.,1.2],]) * self.surface + origin


    def check_and_move(self, near_positions, box, r_cut, pbc=[True, True, False], count_limit= 10000):
        is_good = False
        flip_check = False
        tmp = []
        counts = 0
        # initial check to see if anything is too close
        for p in self.positions:
            dists = distance(p, near_positions, box, pbc)
            tmp.append(dists < r_cut)
        num_close = np.array(tmp).flatten().sum()
        if num_close > 0:
            while counts < count_limit:
                v = np.array([random.random() for _ in range(3)])
                for i,p in enumerate(self.positions):
                    self.positions[i] = rotate_about_vec(p - self.origin, v, np.pi/15.) + self.origin
                is_close = []
                for p in self.positions:
                    dists = distance(p, near_positions, box, pbc)
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

def find_surface_atoms(infile, box, pbc):
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
    pipeline.modifiers.append(ConstructSurfaceModifier(radius = 6,
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

def distance(x0, x1, box, pbc=[True, True, False]):
    # xo is a position of one atom, x1 is an array of positions
    # use the pbc bool mask to set the periodicity
    delta = np.abs(x0 - x1)
    delta[:,pbc] -= box[pbc] * np.round(delta[:,pbc]/(box[pbc]))
    return np.sqrt((delta ** 2).sum(axis=-1))

def rotate_about_vec(u,a,theta):
    import quaternion
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
    parser.add_argument('--cov_fact',dest='cov_fact',default=0.5, type=float,
                        help='Coverage factor of oxygen groups on surface, float between 0.0 -> 1.0')
    parser.add_argument('--sep',dest='sep',default=1.8, type=float,
                        help='minimum separation distance between inserted particles and all others, default is 1.8 Å')
    parser.add_argument('--count_limit',dest='count_limit',default=10000, type=int,
                        help='Change the random number seed, default is 666')
    parser.add_argument('--silent', dest='silent', default=False, action='store_true',
                       help="Run program silently")
    parser.add_argument('--surface_atomtype', default='C1', type=str,
                       help="comma separated list of edge carbon atom types, default is C1")
    parser.add_argument('--hydroxyl_epoxy_ratio', default=0.7, type=list,
                       help="probability of hydroxyl group installation out of 1, all other instances will install epoxy groups")
    parser.add_argument('--pbc', default=[True, True, False], type=list,
                        help="periodic boundary conditions, by default there no periodicity in z, i.e. is set to: [True, True, False]")
    parser.add_argument('--check_distances', dest='check_distances', default=False, action='store_true',
                       help="Double check inter-atomic distances after additional atoms have been installed. Default is False.")
    parser.add_argument('--outfile', dest='outfile', default=None, type=str,
                       help="Specify name of output data file, default is the inputfilename with 'MOD.xyz' appended instead of the existing extension.")

    args = parser.parse_args()

    # assign inputs to some normal vars
    infile = args.infile
    r_cut = args.r_cut
    seed = args.seed
    count_limit = args.count_limit
    surface_atype = args.surface_atomtype
    cov_fact = args.cov_fact
    oh2o_ratio = args.hydroxyl_epoxy_ratio
    silent = args.silent
    sep = args.sep
    pbc = args.pbc
    check_dists = args.check_distances
    outfile = args.outfile

    if not silent:
        print(f'Coverage Factor: {cov_fact}')

    # seed is default 666
    random.seed(seed)

    if not outfile: outfile = infile.split('.')[0] + '_MOD.xyz'
    if not silent: print(f'Reading input from file {infile}\n')
    if not silent: print(f'Surface carbons are type {surface_atype}')

    # get data from xyz file
    natoms, box, aids, atypes, atypelabels, apos, unique_atypes, unique_atypelabels = io.read_xyz(infile)
    # set cell origin so that we can use the easy cell boundary condition
    #apos[:,pbc] -= box[pbc] * 0.5
    apos[:,pbc] -= box[pbc] * np.round(apos[:,pbc]/(box[pbc]))
    natomtypes = len(unique_atypes)
    print(natoms, box)

    # randomly select C1 surface atoms for functionalisation
    surface_mask = atypelabels == surface_atype
    chance = np.array([random.random() for _ in range(natoms)])
    mask = np.logical_and( surface_mask, chance <= cov_fact)
    idx = np.where(mask)[0]
    # initialise things, 'added_groups' is a list of group objects that are to be installed
    added_groups = []
    denied_hydroxyl = 0
    denied_epoxy = 0
    p = progressbar(25)
    p.updateinitial('installing groups','-+')
    # iterate over selected surface target carbons
    for i,(c_id, c_pos, c_label, c_idx) in enumerate(zip(aids[mask], apos[mask], atypelabels[mask], idx)):
        num_sites = mask.sum()
        # update the progress bar
        percentage = float(i)/float(num_sites)
        p.update(percentage,'installing groups (%d/%d)'%(i+1,num_sites))
        # add the new atom positions so they can be avoided when further adding groups groups
        if len(added_groups)>0:
            n = len(added_groups)
            all_pos = np.concatenate((apos, np.row_stack([g.positions for g in added_groups])))
        else:
            all_pos = apos

        # find neighbour carbons bonded to target carbon
        dists = distance(c_pos, apos, box, pbc= pbc)
        bonded_mask = np.logical_and(dists < r_cut, dists > 0.01)
        bonded_pos = apos[bonded_mask]
        bonded_labels = atypelabels[bonded_mask]
        bonded_ids = aids[bonded_mask]

        # find ALL atoms withn 4A of the target carbon
        dists = distance(c_pos, all_pos, box, pbc=pbc)
        all_dist_mask = (dists < 3.2) & (dists > 0.01)
        near_pos = all_pos[all_dist_mask].copy()

        # roll the dice to decide which group is to be installed
        rand = random.random()
        # hydroxyl
        if rand < oh2o_ratio:
            g = group('hydroxyl', c_pos, box)
            status = g.check_and_move(near_pos, box, sep, count_limit= count_limit)
            if status == True:
                added_groups.append(g)
                atypelabels[c_idx] = 'C3'
            else:
                denied_hydroxyl += 1

        # epoxy
        elif rand > oh2o_ratio:
            #try to attach epoxy group iterating over the bonded neighbours of the target carbon, stop if we succeed
            for b_idx in range(3):
                # check that both site carbons of the potential epoxy site are not adjacent to another epoxy carbon
                dists = distance(bonded_pos[b_idx], apos, box, pbc= pbc)
                nextto_bonded = (dists < r_cut) & (dists > 0.01)
                nextto_labels = atypelabels[nextto_bonded]
                tmp = np.concatenate((nextto_labels,bonded_labels))
                if 'C4' not in tmp:
                    # check that there is nothing bonded to the other site carbon
                    if (bonded_labels[b_idx] == 'C1'):
                        # get vector between the two carbon bonding sites
                        tmp = bonded_pos[b_idx] - c_pos
                        tmp[pbc] -= box[pbc]  * np.round(tmp[pbc]/box[pbc])
                        origin = tmp * 0.499 + c_pos
                        # the origin of the epoxy is the midpoint of this vector
                        origin[pbc] -= box[pbc] * np.round(origin[pbc]/box[pbc])
                        g = group('epoxy', origin, box)
                        # for epoxy, remove the other bonded carbon from the array of near positions
                        status = g.check_and_move(near_pos[[not np.array_equal(bonded_pos[b_idx], pp) for pp in near_pos]],
                                                  box,
                                                  sep,
                                                  count_limit= count_limit)
                        if status == True:
                            if (c_idx == 17924) or (c_idx == 42383): pdb.set_trace()
                            added_groups.append(g)
                            atypelabels[bonded_ids[b_idx] == aids] = 'C4'
                            atypelabels[c_idx] = 'C4'
                            break
            if status == False:
                denied_epoxy += 1
        if i+1==num_sites: p.updatefinal('install groups (%d/%d)'%(num_sites,num_sites))


    pdb.set_trace()
    if len(added_groups) > 0:
        apos = np.row_stack((apos, np.row_stack([g.positions for g in added_groups])))
        atypelabels = np.concatenate((atypelabels, np.array([el for g in added_groups for el in g.labels], dtype=str)))

    for i, ipos in enumerate(apos[:-1]):
        dists = distance(ipos, apos[i+1:], box, pbc= pbc)
        if np.any(dists < sep):
            j = np.where(dists<0.8)[0] + i+1
            for jj in j:
                print('Problem!!')
                print(f'atoms {i} and {jj} are too close, {atypelabels[i]}-{atypelabels[jj]}')

    pdb.set_trace()
    newbox = box.copy()
    natoms += int(np.array([g.natoms for g in added_groups]).sum())
    #apos[:,pbc] -= box[pbc] * np.round(apos[:,pbc]/box[pbc])
    apos[:,pbc] += box[pbc] * 0.5
    apos[:,-1] += 20.0
    min_pos = np.min(apos[:,-1]) - 4
    apos[:,-1] -= min_pos
    newbox[2] = np.max(apos[:,2]) + 40.0

    print(f'{denied_hydroxyl} hydroxyls and {denied_epoxy} epoxies failed to install')

    print('Writing result to file')
    io.write_xyz(natoms, atypelabels, apos, newbox, outfile)

    print('Writing log file')
    with open('log.txt','w') as fo:
        fo.write(f'Coverage factor: {cov_fact}\n')
        fo.write(f'Hydroxyl probability: {oh2o_ratio}, epoxy: {1. - oh2o_ratio}\n')
        fo.write(f'Fitting attempts per group: {count_limit}\n')
        types, counts = np.unique(np.array([g.group for g in added_groups]), return_counts= True)
        for tt, num in zip (types, counts):
            fo.write(f'{num} {tt}\n')
        for i,g in enumerate(added_groups):
            fo.write(f'{i+1} {g.group} {g.surface}\n')
