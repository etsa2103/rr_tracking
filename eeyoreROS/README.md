## EEYORE-ROS
This is a ROS1 package that incorporates the the Boson and Spinnaker SDKs from FLIR in one near package. Use it for one camera or both. Note that is packer is dependent on [Eeyore](https://github.com/jhughes50/eeyore) which is non-ROS wrapper around both SDKs, if you do not need ROS use this package instead. I named this packer eeyore because I was tired of saying E.O.I.R. 

### Notes
This package only works on debian based linux distributions and requires udev rules to be installed on your host machine (even if you use the docker image). The original version of Boson SDK does not support these udev my custom usev rules, so you will have to use the boson SDK [here](https://github.com/jhughes50/boson-sdk). Note that this installs two udev rules, one for hardware triggering on the boson as part of the `ACM` subsystem and one for the image stream and part of the `video4linux` subsystem.

### Setup
The only thing to setup before building an running is installing the udev rules. They can be copied using  `cp ./etc/*.rules /etc/udev/rules.d/`. The Spinnaker SDK needs to be downloaded from [here](https://www.flir.com/products/spinnaker-sdk/?vertical=machine+vision&segment=iis). Extract the files and move them to `./install/spinnaker-3.2.0.57-amd64` for the sdk inself and `./install/spinnaker_python` for PySpin. The docker build system will copy the files from these directories respectively, if you are not using docker simply install the SDK following their instructions.

### Docker
To build the docker image navigate to `./Docker` and run `./build.bash`. Once the image is built run it with `./run.bash`.
