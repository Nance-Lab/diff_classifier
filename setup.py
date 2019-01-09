from setuptools import setup, find_packages
import setuptools.command.build_py
import os
# try:
#    import fijibin
# except:
#    print('Import error. Install Fiji manually.')

ver_file = os.path.join('diff_classifier', 'version.py')
with open(ver_file) as f:
    exec(f.read())

PACKAGES = find_packages()

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            packages=PACKAGES,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            install_requires=REQUIRES,
            package_data=PACKAGE_DATA)

class FijiCommand(setuptools.command.build_py.build_py):
    """Downloads and unzips current version of Fiji."""
    
    def run(self):
        self.run_command('cd ~')
        self.run_command('wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip')
        self.run_command('unzip fiji-linux64.zip')
        setuptools.command.build_py.build_py.run(self)


if __name__ == '__main__':
    setup(**opts)
    try:
        #import fijibin
        setup(cmdclass={'build_py': FijiCommand})
    except:
        print('Import error. Install Fiji manually.')
