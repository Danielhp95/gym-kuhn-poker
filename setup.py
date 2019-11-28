from setuptools import setup, find_packages

setup(name='gym_kuhn_poker',
      version='0.1',
      description='OpenAI gym environment for Kuhn poker',
      url='https://github.com/Danielhp95/gym-kuhn-poker',
      author='Sarios',
      author_email='madness@xcape.com',
      packages=find_packages(),
      install_requires=['gym', 'numpy']
      )
