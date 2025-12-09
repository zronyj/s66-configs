# README

## How to create the database

### 1. Download
The S66 configurations database is built by using the S66 database, which in turn comes from the NENCI-2021 database.
The latter can be downloaded from [here](https://materials.colabfit.org/dataset-original/DS_0j2smy6relq0_0).

### 2. Extract
Please the compressed file into this folder, and run the script `build_database.sh`. This will create a folder containing all the files from the original NENCI-2021 database, and it will also create a folder containing the files of the S66 database.

It should be noted that the extracted database differs from the one described in the [Supporting Information](https://pubs.acs.org/doi/suppl/10.1021/ct2002946/suppl_file/ct2002946_si_001.pdf) from the article by some names, but not its content.
The differences are the following:
- the entries 35, 26, 37 and 38 from the SI correspond to the entries 36, 38, 35, 37 in the current database
- entry 47 of the SI has been moved to position 66 in the database, shifting all entries starting at 48 in the SI to a lower entry in the database (e.g. 48 -> 47, 49 -> 48, 50 -> 49).

### 3. Run constrained MD on all of them

The Molecular Dynamics run is done with the **[SO3LR](https://pubs.acs.org/doi/10.1021/jacs.5c09558)** force field. You can install it by following the instructions listed in its [GitHub Repository](https://github.com/general-molecular-simulations/so3lr).

Once you have a working Python environment with **SO3LR** available as a library through ASE, you may go to the `md` directory and run the script `md_runner.py`. Please note that this will trigger the execution of 462 MD simulations in serial. If your **SO3LR** installation can use a GPU, the latter should take around 8 hours.

Once the MD runs are finished, you should find `*.log` and `*.traj` files for every system in the S66 database.

### 4. Extracting XYZ data and aligning the dimers

After the MD runs are finished, you can switch to the `aligned` folder. The latter contains the `traj_reader.py` script. Run this to obtain the trajectories as XYZ files, where the centers of mass -COM- of all dimers are aligned to the same axis. The resulting files will appear in the folder as `aligned_*.xyz` files which are already human readable and can be opened with software such as **VMD**.

### 5. Extract 10 unique structures from each trajectory

After translating the trajectories into XYZ files, you may go to the `filtered` folder and run the `struct_filter.py` script. The latter will attempt to extract 10 unique conformations of the dimers based on their energy distribution and RMSD value.

You can change the parameters of the filter by editing the script. At the very end, there is a description of all the parameters that can be changed or selected. But please bear in mind that the *energy* method is quite fast compared to the *rmsd* method. The latter takes around 30 seconds per trajectory on a modern laptop.

The unique configurations for each entry in the S66 database are, in this case, specific frames of each trajectory listed in each `filtered*.txt`.

## How to get the databases

All the previous steps have been carried out successfully and packed in `*.tar.bz2` files in this repository.

The last script, to extract these frames and save them as individual XYZ files will be included in here shortly.