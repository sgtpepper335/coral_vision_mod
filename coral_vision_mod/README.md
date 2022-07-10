# edgetpuvision

Python API to run inference on image data coming from the camera.

## Build

python3 setup.py sdist
python3 setup.py bdist
python3 setup.py sdist_wheel

## Debian pacakge

To build debian pacakge run:
```
dpkg-buildpackage -b -rfakeroot -us -uc -tc
```
