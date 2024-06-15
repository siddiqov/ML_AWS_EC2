from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns a list of requirements.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='ML_AWS_EC2',
    version='0.0.1',
    author='Hafeez',
    author_email='siddiqov@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')  # ensure the file name matches
)
