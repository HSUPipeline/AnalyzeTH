"""Run TH analysis across all sessions."""

from pynwb import NWBHDF5IO

from settings import DATA_PATH, REPORTS_PATH, RESULTS_PATH, IGNORE

# Import local code
# import sys
# sys.path.append('../code')
# from ... import ...

###################################################################################################
###################################################################################################

def main():
    """Run session analyses."""

    nwbfiles = get_files(DATA_PATH, select='nwb')

    for nwbfile in nwbfiles:

        # Check and ignore files set to ignore
        if nwbfile.split('.')[0] in IGNORE:
            print('Skipping file: ', nwbfile)
            continue

        # Load file and prepare data
        print('Running session analysis: ', nwbfile)

        # Get subject name & load NWB file
        nwbfile = NWBHDF5IO(str(DATA_PATH / nwbfile), 'r').read()

        # Get the subject & session ID from file
        subj_id = nwbfile.subject.subject_id
        session_id = nwbfile.session_id

        # Get data of interest from the NWB file
        ...

        ## ANALYZE SESSION DATA
        ...

        ## CREATE REPORT
        ...


if __name__ == '__main__':
    main()
