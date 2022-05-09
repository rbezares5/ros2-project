from setuptools import setup

package_name = 'robotx'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Raul Bezares',
    maintainer_email='rbezares5@uma.es',
    description='TODO: Package description',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'application = robotx.directorNode:main',
            'calibration = robotx.calibratorNode:main',
            'vision = robotx.computerVisionNode:main',
            'agent = robotx.checkersAgentNode:main',
            'human = robotx.humanPlayerNode:main',
            'robot = robotx.robotNode:main',
        ],
    },
)
