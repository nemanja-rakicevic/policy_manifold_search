
import os

from setuptools import setup, find_packages, Command


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='PolicyManifoldSearch',
    url='anonymous',
    author='anonymous',
    author_email='anonymous',
    # Needed to actually package something
    # packages=[package for package in find_packages() if package.startswith('behaviour_representations')],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Code for Policy Manifold Search algorithm',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
    cmdclass={
        'clean': CleanCommand,
    },
    extra_link_args=['-L/usr/lib/x86_64-linux-gnu/']
)
