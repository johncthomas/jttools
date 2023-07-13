import os, pickle, logging
import pathlib
import datetime

import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)

class PicklePot:
    """
    For managing pickled objects in a directory. Only works in the same scope from which
    it is initialised.

    Written pickle files have a version number.
    """

    def __init__(self, pickle_dir='pickles', potname='picklepot',):
        """
        Initializes a PicklePot object with a directory path.

        Calls self.load_pickles()

        Parameters:
            pickle_dir (str): The directory path to load and dump pickled objects.
        """
        os.makedirs(pickle_dir, exist_ok=True)
        self.pickle_dir = pickle_dir
        self.objects = {}
        self.load_pickles()
        self.print_assign_strings(potname)
        self.potname = potname
        self._ledger_path = pathlib.Path(self.pickle_dir)/f"picklepot_ledger.csv"


    def print_assign_strings(self, potname='picklepot'):
        # for k in self.objects:
        #     print(f"{k} = {instance_name}.objects['{k}']")
        s = f"""
# to move picklepot objects to global...
exclude = []
for k in {potname}.objects:
    if k not in exclude:
        exec(f"{{k}} = {potname}.objects['{{k}}']")
"""
        print(s)

    def _get_latest_versions(self, files=None) -> dict[str, int]:
        """Use self.pickle_dir by default"""
        if files is None:
            files = os.listdir(self.pickle_dir)
        versions = {}
        for file in files:
            if file.endswith('.pickle'):
                obj_name, version = os.path.splitext(file)[0].split('.')
                version = int(version)
                if (obj_name not in versions) or (version > versions[obj_name]):
                    versions[obj_name] = version
        return versions

    def _write_to_ledger(self, objname, ver, info):
        ver = str(ver)

        with open(self._ledger_path, 'a') as f:
            d = datetime.datetime.now()
            now = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}"

            f.write(
                ','.join(
                    [objname, ver, info, now]
                ) +'\n'
            )

    def ledger(self, latest_only=False) -> pd.DataFrame:
        """A DataFrame of files written to the picklepot.

        :parameter
            lastest: When True, only the last written version for each
                object name is added to the table."""
        ledger = pd.read_csv(self._ledger_path, header=None)
        ledger.columns = ['Name', 'Version', 'Info', 'Date']
        if latest_only:
            return ledger.loc[ledger.Name.duplicated(keep='last')]
        return ledger.loc[ledger.duplicated(['Name', 'Version'], keep='last')]


    def version_history(self, objname, print_it=True) -> pd.DataFrame:
        ledger = self.ledger()
        ledger:pd.DataFrame = ledger.loc[ledger.Name == objname]
        if print_it:
            for _, row in ledger.iterrows():
                print(
                    f"{row.Version} ({row.Date}):  {row.Info}"
                )
        return ledger


    def versions(self) -> dict[str, int]:
        return self._get_latest_versions()

    def load_pickles(self):
        """
        Load the latest version of pickles in PicklePot.pickle_dir to self.objects.
        """

        for obj_name, version in self.versions().items():

            self.objects[obj_name] = self.load_obj(obj_name, version)
        print('\n')


    def dump(self, obj, obj_name, info='', overwrite_latest=False):
        """
        Writes a pickled object to a file in the pickle_dir directory.
        If the object has already been pickled, the version number is incremented.

        Parameters:
            obj: A picklable object
            obj_name (str): The name AS STRING of the object to pickle.
            info: String describing the object, or differences from previous versions.
            overwrite_latest: don't incriment the version number, overwrite highest
                version pickle file.
        """
        versions = self.versions()
        if obj_name in versions:
            version = versions[obj_name]
            if not overwrite_latest:
                version += 1

            file_name = os.path.join(self.pickle_dir, f"{obj_name}.{version}.pickle")
        else:
            version = 1
            file_name = os.path.join(self.pickle_dir, f"{obj_name}.{version}.pickle")

        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)
            self._write_to_ledger(obj_name, version, info)

        print(f"Saved {obj_name} (version {version})")


    def load_obj(self, obj_name, version=0, source_dir=None):
        """
        Load a specific object. By default, the latest version is loaded, use `version=n` to get a specific version.

        Parameters:
            obj_name (str): The name of the object to load.
            version (int): The version number of the object to load. Defaults to 0, which loads the latest version.
            source_dir: Source dir, default is self.pickle_dir

        Returns:
            The loaded object.
        """
        if source_dir is None:
            source_dir = self.pickle_dir
        if version == 0:
            try:
                version = self._get_latest_versions(
                    source_dir
                )[obj_name]
            except KeyError:
                raise FileNotFoundError(f"No files matching {obj_name}.n.pickle found in {source_dir}")
        file_name = os.path.join(source_dir, f"{obj_name}.{version}.pickle")
        try:
            with open(file_name, 'rb') as f:
                obj = pickle.load(f)
                print(f"Loaded {obj_name} (version {version})")
                return obj
        except Exception as e:
            logger.warning(
                f"Failed to unpickle {file_name}.\n{str(e)}"
            )

    def __getitem__(self, item):
        return self.objects[item]



def _test_picklepot(tmpdir='/home/jthomas/tmp/pickles/'):
    tmpdir = pathlib.Path(tmpdir)
    i0 = 'xxx'
    i1 = 'aaa'
    i2 = 'fff'
    p = PicklePot(tmpdir, potname='fart')
    p.dump(i0, 'o1', 'v0')
    p.dump(i1, 'o1', 'v1')
    p.dump(i1, 'o1', 'v1 overwrit', overwrite_latest=True)
    p.dump(i2, 'o2')
    p2 = PicklePot(tmpdir, potname='fart2')

    # can load specific ver
    assert  i0 == p.load_obj('o1', version=1)
    # auto load objects match
    assert p2['o1'] == i1
    assert p2['o2'] == i2

    print('ledger:', p.ledger())
    print('ledger, latest only', p.ledger(latest_only=True))
    print('ver history:', p.version_history('o1', print_it=True))
    import glob
    for fn in  glob.glob(str(tmpdir/'*.pickle')):
        os.remove(fn)
    os.remove(tmpdir/p._ledger_path)

    os.rmdir(tmpdir)

if __name__ == '__main__':
    _test_picklepot()


