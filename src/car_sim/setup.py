import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'car_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share',package_name, 'urdf'),
            glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), 
            glob('map/*.sdf')),
        (os.path.join('share', package_name, 'param'),
            glob('param/*.yaml')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jegur',
    maintainer_email='jegur@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_sensor_test = camera.camera_sensor_test:main',
            'camera_sensor = camera.camera_sensor:main',  # 실제 제어 노드 추가
            'camera_test = camera.camera_test:main',  # 테스트용 노드 추가
        ],
    },
)
