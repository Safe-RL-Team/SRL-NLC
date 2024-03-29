from setuptools import setup

setup(
    name='gym_minigrid',
    version='1.0.1',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    description='Minimalistic gridworld package for OpenAI Gym',
    packages=['gym_minigrid', 'gym_minigrid.envs'],
    package_data={'gym_minigrid': ['data/*.json']},
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ],
    include_package_data=True
)
