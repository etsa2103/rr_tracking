FROM ubuntu:focal

#Run the frontend first so it doesn't throw an error later
RUN apt-get update \
 && export TZ="America/New_York" \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
 && ln -fs "/usr/share/zoneinfo/$TZ" /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && apt-get clean

# General dependencies for development
RUN apt-get update \
 && apt-get install -y \
        build-essential \
        cmake \
        cppcheck \
        gdb \
        git \
        libeigen3-dev \
        g++ \
        libbluetooth-dev \
        libcwiid-dev \
        libgoogle-glog-dev \
        libspnav-dev \
        libusb-dev \
        lsb-release \
        mercurial \
        python3-dbg \
        python3-empy \
        python3-pip \
        python3-venv \
        software-properties-common \
        sudo \
        wget \
	    curl \
        cmake-curses-gui \
        geany \
        tmux \
        dbus-x11 \
        iputils-ping \
        default-jre \
        iproute2 \
 && apt-get clean

ARG user_id=1000
env USER eeyore
RUN useradd -U --uid ${user_id} -ms /bin/bash $USER \
 && echo "$USER:$USER" | chpasswd \
 && adduser $USER sudo \
 && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER

USER $USER

RUN mkdir -p /home/$USER/install
WORKDIR /home/$USER
COPY install /home/$USER/install/

# Install ROS Noetic
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \ && sudo /bin/sh -c 'curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -' \
 && sudo apt-get update \
 && sudo apt-get install -y \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    ros-noetic-desktop-full

RUN sudo rosdep init \
 && sudo apt-get clean

RUN rosdep update

# setup groups
RUN sudo addgroup flirimaging && sudo adduser $USER flirimaging 
RUN sudo adduser $USER dialout
RUN sudo adduser $USER tty 
RUN sudo adduser $USER plugdev
RUN sudo adduser $USER video

# install pyspin dependencies
RUN sudo apt-get update \
 && sudo apt-get install -y libusb-1.0-0 \
 && sudo apt-get update \
 && sudo apt-get install -y libswscale-dev \
 && sudo apt-get update \
 && sudo apt-get install -y ffmpeg \
 && sudo apt-get install -y libavcodec-dev

#install spinnaker sdk
RUN cd ~/install/spinnaker-3.2.0.57-amd64 \
 && sudo apt-get update \
 && yes yes | DEBIAN_FRONTEND=noninteractive sudo dpkg -i libgentl_*.deb \
 && sudo dpkg -i libspinnaker_*.deb \
 && sudo dpkg -i libspinnaker-dev_*.deb \
 && sudo dpkg -i libspinnaker-c_*.deb \
 && sudo dpkg -i libspinnaker-c-dev_*.deb \
 && sudo dpkg -i libspinvideo_*.deb \
 && sudo dpkg -i libspinvideo-dev_*.deb \
 && sudo dpkg -i libspinvideo-c_*.deb \
 && sudo dpkg -i libspinvideo-c-dev_*.deb \
 && sudo apt-get install -y ./spinview-qt_*.deb \
 && sudo dpkg -i spinview-qt-dev_*.deb \
 && sudo dpkg -i spinupdate_*.deb \
 && sudo dpkg -i spinupdate-dev_*.deb \
 && sudo dpkg -i spinnaker_*.deb \
 && sudo dpkg -i spinnaker-doc_*.deb \
 && sudo bash configure_usbfs.sh \
 && sudo bash configure_gentl_paths.sh 64 \
 && sudo bash configure_spinnaker_paths.sh

#install pyspin
RUN pip3 install --upgrade numpy matplotlib \
 && pip3 install Pillow==9.2.0 \
 && cd ~/install/spinnaker_python \
 && pip3 install spinnaker_python-3.2.0.57-cp38-cp38-linux_x86_64.whl 

#install boson sdk 
RUN cd ~/install \
 && git clone https://github.com/jhughes50/boson-sdk \
 && cd boson-sdk \
 && sudo bash buildAndInstall.sh

#make a workspace
RUN mkdir -p /home/$USER/ws/src \
 && sudo chown $USER:$USER -R /home/$USER/ws

#install eeyore
RUN cd /home/$USER/ws/src \
 && git clone https://github.com/jhughes50/eeyore

#copy over eeyoreROS
RUN mkdir -p /home/$USER/ws/src/eeyore_ros
COPY ../ /home/$USER/ws/src/eeyore_ros

RUN /bin/sh -c 'echo sudo chown $USER:$USER ~/.ros >> ~/.bashrc'

# build
RUN cd /home/$USER/ws \
 && sudo chown $USER:$USER -R ~/.config \
 && sudo chown $USER:$USER -R ~/.ros \
 && catkin config --extend /opt/ros/noetic \
 && catkin build --no-status -DCMAKE_BUILD_TYPE=Release

RUN /bin/sh -c 'echo ". /etc/profile.d/setup_spinnaker_paths.sh" >> ~/.bashrc'
RUN /bin/sh -c 'echo ". /opt/ros/noetic/setup.bash" >> ~/.bashrc'
RUN /bin/sh -c 'echo ". ~/ws/devel/setup.bash" >> ~/.bashrc'

# add a folder for data
RUN mkdir -p ~/data
