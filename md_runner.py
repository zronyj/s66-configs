import os
import warnings
import numpy as np

from so3lr import So3lrCalculator
from ase import Atoms
from ase.constraints import FixSubsetCom
from ase.neighborlist import natural_cutoffs, NeighborList

from ase.io import read

from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.md import MDLogger

warnings.filterwarnings("ignore")

def split_into_molecules(atoms: Atoms, mult: float = 1.1):
    """
    Split an ASE Atoms object into molecular fragments using
    covalent-radius-based connectivity.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (can contain many molecules).
    mult : float
        Global scaling of covalent radii. Increase slightly if
        bonds are missed; decrease if different molecules get
        spuriously connected.

    Returns
    -------
    components : list of list of int
        List of the indices of the atoms in each molecule.
    """
    # Per-atom cutoffs from covalent radii
    cutoffs = natural_cutoffs(atoms, mult=mult)

    # Build neighbor list: two atoms are neighbors if their
    # covalent-radius spheres overlap.
    nl = NeighborList(cutoffs,
                      self_interaction=False,
                      bothways=True)
    nl.update(atoms)

    # Build adjacency list (graph of bonds)
    n = len(atoms)
    adj = [[] for _ in range(n)]
    for i in range(n):
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            adj[i].append(j)

    # Find connected components by DFS/BFS
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in adj[i]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
        components.append(sorted(comp))

    return components

def get_centers_of_mass(atoms: Atoms, components: list):
    """
    Calculate centers of mass for molecular fragments.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (can contain many molecules).
    components : list of list of int
        List of the indices of the atoms in each molecule.

    Returns
    -------
    coms : list of np.ndarray
        List of center-of-mass coordinates for each molecule.
    """
    coms = []
    for comp in components:
        sub_atoms = atoms[comp]
        mass = sub_atoms.get_masses().sum()
        com = np.sum(sub_atoms.get_positions() * sub_atoms.get_masses()[:, np.newaxis], axis=0) / mass
        coms.append(com)
    return coms

def get_coms(atoms: Atoms, positions: np.ndarray, components: list):
    """
    Calculate centers of mass for molecular fragments
    at given positions.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (can contain many molecules).
    positions : np.ndarray
        Positions to use for the calculation.
    components : list of list of int
        List of the indices of the atoms in each molecule.

    Returns
    -------
    coms : list of np.ndarray
        List of center-of-mass coordinates for each molecule.
    """
    coms = []
    masses = atoms.get_masses()
    for comp in components:
        mass = masses[comp].sum()
        com = np.sum(positions[comp] * masses[comp][:, np.newaxis], axis=0) / mass
        coms.append(com)
    return coms

class FixCOMs():
    """
    Constraint to fix the center of mass of all molecules
    defined by a list of lists of atom indices.
    """
    def __init__(self, indices, coms):
        self.idxs = indices
        self.coms = coms
    
    def index_shuffle(self, atoms: Atoms, ind):
        # No shuffling needed
        pass
    
    def get_removed_dof(self, atoms):
        # 3 degrees of freedom per molecule
        return 3 * len(self.idxs)

    def adjust_positions(self, atoms, newpositions):
        """
        This method adjusts the positions to fix the centers of mass.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The Atoms object.
        newpositions : np.ndarray
            The new positions to be adjusted.
        
        Notes
        -----
        The positions to be updated are provided in `newpositions`.
        The method modifies `newpositions` in place to ensure that
        the centers of mass of the specified groups of atoms remain fixed.
        """
        # Get current centers of mass
        current_coms = get_coms(atoms, newpositions, self.idxs)

        # print("Adjusting COMs:", np.linalg.norm(np.array(current_coms) - np.array(self.coms), axis=1))

        # Calculate the difference between the current and desired COMs
        diffs = []
        for i, idx in enumerate(self.idxs):
            diff = current_coms[i] - self.coms[i]
            diffs.append(diff)

            # Update the positions
            for a in idx:
                newpositions[a] -= diff

    def adjust_forces(self, atoms, forces):
        # No forces to adjust
        pass

    def todict(self):
        # Return a dictionary representation of the FixCOMs object
        return {'name': 'FixCOMs',
                'kwargs': {'indices': self.idxs,
                           'coms': self.coms}}

if __name__ == '__main__':

    # Get the current directory
    here = os.path.dirname(os.path.abspath(__file__))

    # Get the path of the XYZ files
    xyz_path = os.path.join(here, '..', 's66')
    xyz_dimers = [f for f in os.listdir(xyz_path) if f.endswith('.xyz')]
    xyz_dimers.sort()

    # Get already existing trajectories
    traj_dimers = ['_'.join(f.split('_')[1:-1]) + '.xyz' for f in os.listdir(here) if f.endswith('.xyz')]
    traj_dimers.sort()

    nb_dimers = len(xyz_dimers)

    print(f'Found {nb_dimers} XYZ files for MD simulations.')

    for d, dimer in enumerate(xyz_dimers):

        if dimer in traj_dimers:
            print(f'\n[{d:0>3}/{nb_dimers}] Skipping {dimer} as trajectory already exists.')
            continue

        print(f'\n[{d:0>3}/{nb_dimers}] Running MD for {dimer}...')

        try:
            os.remove(f'{dimer[:-4]}_md.traj')
            os.remove(f'{dimer[:-4]}_md.log')
        except FileNotFoundError:
            pass

        # Read the dimer structure
        atoms = read(os.path.join(xyz_path, dimer))
        atoms.info['charge'] = 0.0

        # Split into molecules to fix their centers of mass
        molecules = split_into_molecules(atoms, mult=1.1)

        mult = 1.1

        # Ensure that there are two molecules
        # The mult parameter is used to adjust the covalent radii until two molecules are found
        while len(molecules) < 2 and mult > 0.5:
            mult -= 0.05
            molecules = split_into_molecules(atoms, mult=mult)

        if len(molecules) > 2 or mult <= 0.5:
            print(f'- Could not split atoms {dimer} into two molecules.')
            continue

        # Calculate the inter-center-of-mass distance
        coms = get_centers_of_mass(atoms, molecules)

        # constraints = []
        # for m, mol in enumerate(molecules):
            # constraints.append(FixSubsetCom(indices=mol))

        constraints = [FixCOMs(indices=molecules, coms=coms)]

        atoms.set_constraint(constraints)

        # Set up the So3lr calculator
        calc = So3lrCalculator(
            calculate_stress=False,
            lr_cutoff=1000,  # for gas-phase systems
            dtype=np.float64
        )
        
        atoms.calc = calc

        # Initialize velocities corresponding to 300 K
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        # Set up the MD integrator (Langevin dynamics)
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.02)

        # Attach a Trajectory object to write positions
        traj = Trajectory(f'{dimer[:-4]}_md.traj', 'w', atoms)
        dyn.attach(traj.write, interval=10)

        # Prepare log file
        dyn.attach(
            MDLogger(dyn, atoms, f'{dimer[:-4]}_md.log', header=True, stress=False),
            interval=10
        )

        # Run the MD simulation
        dyn.run(steps=10000)

        print(f'- MD simulation for {dimer} completed. Log saved to {dimer[:-4]}_md.log.')