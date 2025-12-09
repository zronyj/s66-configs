#!/usr/bin/env python3
import os
import warnings
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count

import MDAnalysis as mda
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def single_atom_square_distance_matrix(universe, selection='all', atom = 0):
    """
    Compute pairwise square distance between all frames of a Universe for a single atom.

    Parameters
    ----------
    universe : MDAnalysis Universe
        The MDAnalysis Universe containing the trajectory.
    selection : str
        Atom selection string for MDAnalysis.
    atom : int
        Index of the atom within the selection to compute distances for.
    
    Returns
    -------
    atom_sq_dist_matrix : (N, N) array
        Pairwise square distance matrix for the specified atom.
    """
    nb_steps = universe.trajectory.n_frames

    atom_sq_dist_matrix = np.zeros((nb_steps, nb_steps))
    universe.trajectory.rewind()
    for step_i in range(nb_steps):
        universe.trajectory[step_i]
        pos_a = universe.select_atoms(selection).positions[atom]
        for step_j in range(step_i, nb_steps):
            universe.trajectory[step_j]
            pos_b = universe.select_atoms(selection).positions[atom]
            diff = pos_a - pos_b
            sq_dist = np.dot(diff, diff)
            atom_sq_dist_matrix[step_i, step_j] = sq_dist
            atom_sq_dist_matrix[step_j, step_i] = sq_dist  # symmetry
    return atom_sq_dist_matrix

class Structure_Filter:
    """
    Class for filtering structures based on RMSD or energy.
    
    Parameters
    ----------
    xyz_file : str
        Path to the trajectory file.
    log_file : str
        Path to the log file.
    selection : str
        Atom selection string for MDAnalysis.
    filter_method : str
        Method for filtering structures. Can be 'rmsd' or 'energy'.
    output_file : str
        Path to the output file.
    similarity_png : str
        Path to the similarity matrix PNG file.
    energy_png : str
        Path to the energy distribution PNG file.
    create_images : bool
        Whether to create similarity and energy distribution images.
    n_unique : int
        Number of unique structures to keep.
    threshold : float
        Threshold for filtering structures based on RMSD or energy.
    seed : int
        Seed for random number generator."""

    def __init__(self, xyz_file, log_file, selection='all',
                 filter_method='rmsd',
                 output_file="filtered_structures.txt",
                 similarity_png="rmsd_similarity.png",
                 energy_png="energy_distribution.png",
                 create_images=True,
                 n_unique=10,
                 threshold=0.5,
                 seed=42):
        self.xyz_file = xyz_file
        self.log_file = log_file
        self.selection = selection
        self.filter_method = filter_method
        self.output_file = output_file
        self.similarity_png = similarity_png
        self.energy_png = energy_png
        self.create_images = create_images
        self.n_unique = n_unique
        self.threshold = threshold
        self.seed = seed
        self.log = ""

        # Load trajectory
        self.universe = self.__load_trajectory()

        # Read energies
        self.energies = self._read_energies_from_log()
    
    def __load_trajectory(self) -> mda.Universe:
        """
        Load trajectory as a Universe.

        Returns
        -------
        universe : MDAnalysis Universe
        """
        if not os.path.isfile(self.xyz_file):
            raise FileNotFoundError(f"Trajectory file '{self.xyz_file}' not found.")
        if not self.xyz_file.endswith('.xyz'):
            raise ValueError("Currently only XYZ format is supported.")
        u = mda.Universe(self.xyz_file)
        return u
    
    def _read_energies_from_log(self) -> np.ndarray:
        """
        Returns energies as a 1D numpy array of length n_frames,
        ordered consistently with trajectory frames.

        Returns
        -------
        energies : np.ndarray
            Energies as a 1D array.
        """
        if not os.path.isfile(self.log_file):
            raise FileNotFoundError(f"Energies file '{self.log_file}' not found.")
        
        n_frames = self.universe.trajectory.n_frames

        energies = []
        with open(self.log_file) as f:
            data = f.readlines()

        for i, line in enumerate(data):
            if i == 0:
                continue  # skip header
            try:
                energy = float(line.split()[1])
                energies.append(energy)
            except ValueError:
                continue
        energies = np.asarray(energies, dtype=float)
        if len(energies) != n_frames:
            raise ValueError(
                f"Found {len(energies)} energies, but trajectory has {n_frames} frames."
            )
        return energies

    def pairwise_square_distance_matrix(self, n_cores = cpu_count()) -> np.ndarray:
        """
        Compute pairwise square distance between all frames of a Universe'
        DistanceMatrix class. Returns a full square square distance matrix.

        Parameters
        ----------
        n_cores : int
            Number of cores to use for parallelization.

        Returns
        -------
        square_dist_matrix : np.ndarray
            Square distance matrix.
        """
        nb_atoms = len(self.universe.select_atoms(self.selection))
        nb_steps = self.universe.trajectory.n_frames

        square_dist_matrix = np.zeros((nb_steps, nb_steps))

        params = [[self.universe, self.selection, atom] for atom in range(nb_atoms)]

        with Pool(n_cores) as pool:
            results = pool.starmap(single_atom_square_distance_matrix, params)
            square_dist_matrix = np.sum(results, axis=0)

        return square_dist_matrix
    
    def rmsd_between_frames(self, pos_a : int, pos_b : int | list[int]) -> float:
        """
        Compute RMSD between two frames or a frame and multiple frames.

        Parameters
        ----------
        pos_a : int
            Index of the first frame.
        pos_b : int or list of int
            Index or list of indices of the second frame(s).

        Returns
        -------
        rmsd_values : float
            RMSD value between the specified frames.
        """
        self.universe.trajectory.rewind()    # Reset trajectory to the beginning

        # Get coordinates of the first frame
        self.universe.trajectory[pos_a]
        coords_a = self.universe.select_atoms(self.selection).positions

        # Compute RMSD
        # Single frame
        if isinstance(pos_b, int):
            self.universe.trajectory[pos_b]
            coords_b = self.universe.select_atoms(self.selection).positions
            diff = coords_a - coords_b
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            return rmsd
        # Multiple frames
        else:
            rmsd = 0
            for pb in pos_b:
                self.universe.trajectory[pb]
                coords_b = self.universe.select_atoms(self.selection).positions
                diff = coords_a - coords_b
                rmsd_i = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
                rmsd += rmsd_i
            rmsd /= len(pos_b)
            return rmsd
        
    def filter_by_rmsd(self, sd_matrix) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify the n_keep most unique structures using pairwise RMSD and energies.

        Parameters
        ----------
        sd_matrix : (N, N) array
            Pairwise RMSD (not similarity).

        Returns
        -------
        indices : array of int
            Indices of the selected frames (0-based).
        scores : array of float
            Corresponding uniqueness scores.
        """
        N = sd_matrix.shape[0]
        if sd_matrix.shape[1] != N:
            raise ValueError("sd_matrix must be square")

        # RMSD of all other frames as a uniqueness measure.
        # We can include (zero) or exclude; difference is small.
        rmsd = np.sqrt(np.sum(sd_matrix, axis=1) / (N - 1))

        self.log += f"Top uniqueness score: {np.max(rmsd):.4f}, lowest: {np.min(rmsd):.4f}\n"

        # Larger score -> more unique
        idx_sorted = np.argsort(rmsd)[::-1]
        top_idx = idx_sorted[:self.n_unique]
        return top_idx, rmsd[top_idx]

    def filter_by_energy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify the n_keep most unique structures using pairwise RMSD and energies.

        Returns
        -------
        indices : array of int
            Indices of the selected frames (0-based).
        scores : array of float
            Corresponding uniqueness scores.
        """
        # Separate energies into bins
        bin_indices = np.digitize(self.energies, bins=np.linspace(np.min(self.energies), np.max(self.energies), self.n_unique+1)) - 1
        frames_in_bin = {i: [j for j, b in enumerate(bin_indices) if b == i] for i in range(self.n_unique)}

        self.log += f"Number of bins: {len(frames_in_bin)}\n"

        # Randomly select one frame from each bin
        np.random.seed(self.seed)
        selected_frames = []
        similarity_score = []
        for bin_idx, frames in frames_in_bin.items():
            selected_frame = np.random.choice(frames)
            if len(selected_frames) == 0:
                selected_frames.append(selected_frame)
                similarity_score.append(1)
            else:
                # This was a while loop, but changed to for loop to avoid infinite loops
                for _ in range(self.universe.trajectory.n_frames):
                    # Check RMSD against already selected frames
                    rmsd_value = self.rmsd_between_frames(selected_frame, selected_frames)
                    similarity = 1 / (1 + rmsd_value**2)
                    if similarity <= self.threshold:
                        selected_frames.append(selected_frame)
                        similarity_score.append(similarity)
                        break
                    else:
                        selected_frame = np.random.choice(frames)

        return selected_frames, similarity_score
    
    def __rescale_sd_to_similarity(self, sd_matrix):
        """
        Convert square distances to a similarity-like matrix S in [0, 1],
        with S_ii = 1 and smaller similarity for larger square distance.

        Parameters
        ----------
        sd_matrix : (N, N) array
            Pairwise square distance matrix.
        """
        # Avoid division by zero if all frames are identical
        ones = np.ones_like(sd_matrix)
        S = 1 / (ones + sd_matrix)
        return S
    
    def __plot_energy_distribution(self) -> tuple[float, float]:
        """
        Plot energy histogram with Gaussian fit.

        Returns
        -------
        mu : float
            Mean of the Gaussian fit.
        sigma : float
            Standard deviation of the Gaussian fit.
        """
        energies = np.asarray(self.energies)
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        n, bins, patches = ax.hist(energies, bins=10, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black', label='Energy distribution')
        
        # Gaussian fit function
        def gaussian(x, mu, sigma):
            return norm.pdf(x, mu, sigma)
        
        # Fit Gaussian to histogram
        bin_centers = (bins[:-1] + bins[1:]) / 2
        popt, pcov = curve_fit(gaussian, bin_centers, n, p0=[energies.mean(), energies.std()])
        mu_fit, sigma_fit = popt
        
        # Plot fitted Gaussian
        x_fit = np.linspace(energies.min(), energies.max(), 1000)
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu_fit:.3f}, σ={sigma_fit:.3f}')
        
        # Formatting
        ax.set_xlabel('Energy')
        ax.set_ylabel('Density')
        ax.set_title('Energy Distribution with Gaussian Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        fig.tight_layout()
        fig.savefig(self.energy_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.log += f"Energy distribution saved to {self.energy_png}\n"
        self.log += f"Fitted Gaussian: μ = {mu_fit:.4f}, σ = {sigma_fit:.4f}\n"
        return mu_fit, sigma_fit

    def __plot_similarity_matrix(self, sim_matrix) -> None:
        """
        Plot the (triangular) similarity / distance matrix with imshow.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(sim_matrix, origin='lower', cmap='viridis')
        ax.set_xlabel('Frame index')
        ax.set_ylabel('Frame index')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('RMSD-based similarity (1 on diagonal)')
        fig.tight_layout()
        fig.savefig(self.similarity_png, dpi=300)
        plt.close(fig)

    def run(self) -> None:
        """
        Run the structure filter."""
        if self.filter_method == 'rmsd':
            # Compute pairwise square distance matrix
            sd_mat = self.pairwise_square_distance_matrix()

            # Convert to similarity-like matrix with 1 on diag
            sim_mat = self.__rescale_sd_to_similarity(sd_mat)

            if self.create_images:
                # Plot triangular/square matrix via imshow
                self.__plot_similarity_matrix(sim_mat)

            # Identify most unique structures
            unique_idx, unique_scores = self.filter_by_rmsd(sd_mat)
        
        elif self.filter_method == 'energy':

            if self.create_images:
                # Plot energy distribution
                self.__plot_energy_distribution()

            # Create a histogram of the energies, and randomly select
            # one frame from each bin, checking that its RMSD with
            # respect to the selected frames is below a threshold
            unique_idx, unique_scores = self.filter_by_energy()

        else:
            raise ValueError(f"Unknown filter method: {self.filter_method}")

        self.log += "Most unique structures (0-based indices):\n"
        for j, (i, s) in enumerate(zip(unique_idx, unique_scores)):
            self.log += f"{j+1:>2})  frame {i:4d}  score = {s: .4f}  energy = {self.energies[i]: .6f}\n"
        
        with open(self.output_file, 'w') as f:
            f.write(self.log)

if __name__ == "__main__":

    # Get all xyz and log files
    here = os.getcwd()
    aligned_files = os.listdir(os.path.join(here, '..', "aligned"))
    md_files = os.listdir(os.path.join(here, '..', "md"))
    xyz_files = [f for f in aligned_files if f.startswith("aligned_") and f.endswith('.xyz')]
    log_files = [f for f in md_files if f.endswith('.log')]

    # Make sure we have the same number of xyz and log files
    assert len(xyz_files) == len(log_files)

    # Run structure filter
    for xyz, log in tqdm(zip(xyz_files, log_files), total=len(xyz_files)):
        base_name = log[:-7]
        filter = Structure_Filter(os.path.join(here, '..', "aligned", xyz),     # Path of the XYZ trajectory
                                  os.path.join(here, '..', "md", log),          # Path of the log file
                                  selection="all",                              # Atom selection (default: all)
                                  filter_method="energy",                       # Filter method (can be "rmsd" or "energy")
                                  output_file=f"filtered_{base_name}.txt",      # Path of the output file (the one with the results)
                                  similarity_png=f"similarity_{base_name}.png", # Path of the similarity matrix image, if 'rmsd' was selected as filter
                                  energy_png=f"energy_{base_name}.png",         # Path of the energy distribution image, if 'energy' was selected as filter
                                  create_images=False,                          # Whether to create similarity or energy distribution images
                                  n_unique=10,                                  # Number of unique structures to be extracted
                                  threshold=0.5,                                # Threshold for pairwise RMSD (used only if 'energy' was selected as filter)
                                  seed=42)                                      # Random seed
        filter.run()