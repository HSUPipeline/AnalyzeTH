"""Run TH analysis across all units."""

from pynwb import NWBHDF5IO

from settings import (DATA_PATH, REPORTS_PATH, RESULTS_PATH)

# Import local code
# import sys
# sys.path.append('../code')
# from ... import ...

###################################################################################################
###################################################################################################

def main():
    """Run unit analyses."""

    # Get the list of NWB files
    nwbfiles = get_files(DATA_PATH, select='nwb')

    # Get list of already generated run units, & drop file names
    output_files = get_files(REPORTS_PATH / 'results', select='json')
    output_files = [file.split('.')[0] for file in output_files]

    for nwbfilename in nwbfiles:

        ## DATA LOADING

        # Check and ignore files set to ignore
        if nwbfilename.split('.')[0] in IGNORE:
            print('\nSkipping file (set to ignore): ', nwbfilename)
            continue

        # Print out status
        print('\nRunning unit analysis: ', nwbfilename)

        # Get subject name & load NWB file
        nwbfile = NWBHDF5IO(str(DATA_PATH / nwbfilename), 'r').read()

        # Get the subject & session ID from file
        subj_id = nwbfile.subject.subject_id
        session_id = nwbfile.session_id

        # Get information of interest from the NWB file
        ...

        ## ANALYZE UNITS

        # Loop across all units
        for unit_ind in range(len(nwbfile.units)):

            # Initialize output unit file name
            name = session_id + '_U' + str(unit_ind).zfill(2)

            # Check if unit already run
            if SKIP_ALREADY_RUN and name in output_files:
                print('\tskipping unit (already run): \tU{:02d}'.format(unit_ind))
                continue

            print('\trunning unit: \t\t\tU{:02d}'.format(unit_ind))

            results = {}

            ...

            ## MAKE REPORT
            ...



if __name__ == '__main__':
    main()
