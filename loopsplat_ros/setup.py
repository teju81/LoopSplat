from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'loopsplat_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        ('bin', glob('scripts/*.sh')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gaussian_slam = loopsplat_ros.src.entities.gaussian_slam:main',
            'frontend_slam = loopsplat_ros.src.entities.frontend_slam:main',
            'backend_slam = loopsplat_ros.src.entities.backend_slam:main',
            'slam_gui = loopsplat_ros.src.gui.slam_gui:main',
            'loopclosure_detection_test = loopsplat_ros.src.entities.loopclosure_detection_test:main',
        ],
    },
)
